#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_geometry::convex_cut::{ConvexCell, CutPlane, RigidCellMotion};
use molla_math::{Quat, Transform, Vec3};
use molla_solvers::strands::{
    Bond, BondMaterial, Fiber, GpuStrandLimits, GpuStrands, Pin, StepConfig, StrandModel, step_cpu,
};

#[test]
#[ignore = "requires a GPU"]
#[cfg(feature = "resource-counters")]
fn cached_impulses_reuse_only_current_successful_publications() {
    use molla_core::gpu_metrics::TransferScope;
    let engine = Engine::new_gpu().unwrap();
    let mut gpu = GpuStrands::new(&engine, model(3, true, false)).unwrap();
    let initial = gpu.snapshot(&engine).unwrap();
    let impulse = [(1, initial.fiber(1).unwrap().center(), Vec3::X * 0.002)];
    let scope = TransferScope::start().unwrap();
    let applied = gpu
        .impulses_snapshot_cached(&engine, &initial, &impulse)
        .unwrap();
    assert_eq!(scope.finish().readback_bytes, 0);
    compare(&applied, &gpu.snapshot(&engine).unwrap(), 2e-6);
    let mut stale = initial.clone();
    stale.impulse(2, Vec3::ZERO, Vec3::X * 100.0).unwrap();
    let scope = TransferScope::start().unwrap();
    let current = gpu.impulses_snapshot_cached(&engine, &stale, &[]).unwrap();
    assert!(scope.finish().readback_bytes > 0);
    compare(&applied, &current, 2e-6);
    let valid = gpu.snapshot(&engine).unwrap();
    assert!(
        gpu.impulses_snapshot_cached(
            &engine,
            &valid,
            &[(1, Vec3::ZERO, Vec3::X), (999, Vec3::ZERO, Vec3::X)]
        )
        .is_err()
    );
    compare(&valid, &gpu.snapshot(&engine).unwrap(), 1e-10);
    let old = gpu.snapshot(&engine).unwrap();
    gpu.step(
        &engine,
        StepConfig {
            ground_height: None,
            ..Default::default()
        },
    )
    .unwrap();
    let scope = TransferScope::start().unwrap();
    let advanced = gpu.impulses_snapshot_cached(&engine, &old, &[]).unwrap();
    assert!(scope.finish().readback_bytes > 0);
    assert!(
        advanced
            .fiber(2)
            .unwrap()
            .center()
            .distance(old.fiber(2).unwrap().center())
            > 1e-6
    );
}

#[test]
#[ignore = "requires a GPU"]
fn independent_elastic_islands_match_separate_gpu_solves() {
    use molla_solvers::strands::CoupledConfig;
    let engine = Engine::new_gpu().unwrap();
    for sizes in [vec![48, 49, 50], vec![129, 3, 1]] {
        let mut fibers = Vec::new();
        let mut bonds = Vec::new();
        let mut separate = Vec::new();
        for (group, count) in sizes.into_iter().enumerate() {
            let original = model(count, true, false);
            let mut fs = original.fibers().to_vec();
            let mut bs = original.bonds().to_vec();
            for f in &mut fs {
                f.id += group as u64 * 1000;
            }
            for b in &mut bs {
                b.id += group as u64 * 1000;
                b.fibers = b.fibers.map(|id| id + group as u64 * 1000);
            }
            let island = StrandModel::new(fs.clone(), bs.clone()).unwrap();
            separate.push(GpuStrands::new(&engine, island).unwrap());
            fibers.extend(fs);
            bonds.extend(bs);
        }
        fibers.sort_by_key(|f| (f.id % 1000, f.id / 1000));
        let mut batched =
            GpuStrands::new(&engine, StrandModel::new(fibers, bonds).unwrap()).unwrap();
        let config = StepConfig {
            dt: 1.0 / 1000.0,
            ground_height: None,
            coupled: Some(CoupledConfig {
                linear_iterations: 512,
                relative_tolerance: 2e-5,
                ..Default::default()
            }),
            ..Default::default()
        };
        for _ in 0..4 {
            batched.step(&engine, config).unwrap();
            for gpu in &mut separate {
                gpu.step(&engine, config).unwrap();
            }
        }
        let combined = batched.snapshot(&engine).unwrap();
        for gpu in &mut separate {
            let reference = gpu.snapshot(&engine).unwrap();
            for f in reference.fibers() {
                let actual = combined.fiber(f.id).unwrap();
                assert!(actual.center().distance(f.center()) < 2e-6);
                assert!(
                    actual
                        .motion
                        .linear_velocity
                        .distance(f.motion.linear_velocity)
                        < 2e-4
                );
                assert!(
                    actual
                        .motion
                        .angular_velocity
                        .distance(f.motion.angular_velocity)
                        < 2e-4
                );
            }
        }
        assert!(
            batched
                .last_coupled_diagnostics()
                .unwrap()
                .relative_residual
                <= 2e-5
        );
    }
}

#[test]
#[ignore = "requires a GPU"]
fn retained_impulse_batch_matches_cpu_and_is_atomic() {
    let engine = Engine::new_gpu().unwrap();
    let mut cpu = model(3, true, false);
    let mut gpu = GpuStrands::new(&engine, cpu.clone()).unwrap();
    let impulses = vec![
        (
            1,
            cpu.fiber(1).unwrap().center() + Vec3::Y * 0.01,
            Vec3::Z * 0.002,
        ),
        (2, cpu.fiber(2).unwrap().center(), Vec3::X * 0.001),
    ];
    for _ in 0..20 {
        for &(id, p, j) in &impulses {
            cpu.impulse(id, p, j).unwrap();
        }
        let snapshot = gpu.impulses_snapshot(&engine, &impulses).unwrap();
        compare(&cpu, &snapshot, 2e-6);
        compare(&cpu, &gpu.snapshot(&engine).unwrap(), 2e-6);
    }
    let before = gpu.snapshot(&engine).unwrap();
    assert!(
        gpu.impulses(
            &engine,
            &[(1, Vec3::ZERO, Vec3::X), (u64::MAX, Vec3::ZERO, Vec3::X)]
        )
        .is_err()
    );
    compare(&before, &gpu.snapshot(&engine).unwrap(), 1e-10);
    let config = StepConfig {
        gravity: Vec3::ZERO,
        ground_height: None,
        ..Default::default()
    };
    step_cpu(&mut cpu, config).unwrap();
    gpu.step(&engine, config).unwrap();
    compare(&cpu, &gpu.snapshot(&engine).unwrap(), 5e-4);
}

#[cfg(feature = "resource-counters")]
#[test]
#[ignore = "requires a GPU"]
fn retained_impulses_do_not_reallocate_geometry_buffers() {
    use molla_core::gpu_metrics::DeviceScope;
    let engine = Engine::new_gpu().unwrap();
    let mut gpu = GpuStrands::new(&engine, model(3, true, false)).unwrap();
    let scope = DeviceScope::start().unwrap();
    for _ in 0..20 {
        gpu.impulses(&engine, &[(1, Vec3::Y, Vec3::X * 1e-5)])
            .unwrap();
    }
    let counts = scope.finish();
    assert_eq!(counts.observations, 0);
    assert!(!counts.invalid);
}

fn model(count: usize, pinned: bool, damage: bool) -> StrandModel {
    let mut fibers = Vec::new();
    for i in 0..count {
        let rest = Transform::new(
            Vec3::new(i as f64 * 0.3, 1.5, 0.0),
            Quat::from_rotation_y(std::f64::consts::FRAC_PI_2),
        );
        let mut f = Fiber::new(
            i as u64,
            ConvexCell::prism(0.04, 0.3, 8).unwrap(),
            650.0,
            rest,
        )
        .unwrap();
        if pinned && i == 0 {
            f.pin = Some(Pin {
                local_point: Vec3::ZERO,
                pose: rest,
            });
        }
        fibers.push(f);
    }
    let material = BondMaterial {
        translation: Vec3::splat(1e-6),
        rotation: Vec3::splat(0.01),
        force_limit: Vec3::splat(if damage { 0.1 } else { 1e6 }),
        torque_limit: Vec3::splat(if damage { 0.1 } else { 1e6 }),
        damage_rate: if damage { 0.5 } else { 0.0 },
        damping: 0.5,
    };
    let bonds = (1..count)
        .map(|i| {
            Bond::new(
                i as u64,
                [&fibers[i - 1], &fibers[i]],
                [Vec3::Z * 0.15, -Vec3::Z * 0.15],
                material,
            )
            .unwrap()
        })
        .collect();
    StrandModel::new(fibers, bonds).unwrap()
}

fn compare(cpu: &StrandModel, gpu: &StrandModel, tolerance: f64) {
    assert_eq!(cpu.fibers().len(), gpu.fibers().len());
    let mut maximum: f64 = 0.0;
    let mut maximum_velocity: f64 = 0.0;
    for (a, b) in cpu.fibers().iter().zip(gpu.fibers()) {
        maximum = maximum.max(a.center().distance(b.center()));
        maximum_velocity =
            maximum_velocity.max(a.motion.linear_velocity.distance(b.motion.linear_velocity));
        assert!(
            a.center().distance(b.center()) < tolerance,
            "fiber {} CPU {:?} GPU {:?},error {}",
            a.id,
            a.center(),
            b.center(),
            a.center().distance(b.center())
        );
        assert!(a.motion.pose.rotation.dot(b.motion.pose.rotation).abs() > 1.0 - tolerance);
        assert!(a.motion.linear_velocity.distance(b.motion.linear_velocity) < tolerance * 100.0);
    }
    eprintln!("max CPU/GPU center error {maximum}, velocity error {maximum_velocity}");
}

#[test]
#[ignore = "requires a GPU"]
fn slender_free_rotor_preserves_energy_and_matches_cpu() {
    let engine = Engine::new_gpu().unwrap();
    let pose = Transform::new(
        Vec3::Y,
        Quat::from_rotation_x(0.7) * Quat::from_rotation_y(0.3),
    );
    let mut fiber =
        Fiber::new(0, ConvexCell::prism(0.0035, 0.28, 8).unwrap(), 650.0, pose).unwrap();
    fiber.motion.angular_velocity = pose.rotation * Vec3::new(3.0, 2.0, 50.0);
    let invariant = |f: &Fiber| {
        let local = f.motion.pose.rotation.conjugate() * f.motion.angular_velocity;
        let momentum = f.mass.inertia * local;
        (f.motion.pose.rotation * momentum, 0.5 * local.dot(momentum))
    };
    let (momentum, energy) = invariant(&fiber);
    let mut cpu = StrandModel::new(vec![fiber], vec![]).unwrap();
    let mut gpu = GpuStrands::new(&engine, cpu.clone()).unwrap();
    let config = StepConfig {
        gravity: Vec3::ZERO,
        air_damping: 0.0,
        ground_height: None,
        iterations: 1,
        ..Default::default()
    };
    for i in 0..240 {
        step_cpu(&mut cpu, config).unwrap();
        gpu.step(&engine, config).unwrap();
        if i % 24 != 23 {
            continue;
        }
        let actual = gpu.snapshot(&engine).unwrap();
        for (label, f, tolerance) in [
            ("CPU", &cpu.fibers()[0], 1e-8),
            ("GPU", &actual.fibers()[0], 0.003),
        ] {
            let (p, e) = invariant(f);
            assert!(
                (e / energy - 1.0).abs() < tolerance,
                "{label} step {i}: energy ratio {}",
                e / energy
            );
            assert!(
                p.distance(momentum) / momentum.length() < tolerance,
                "{label} step {i}: momentum {p:?}"
            );
        }
        compare(&cpu, &actual, 0.002);
    }
}

#[test]
#[ignore = "dense CPU oracle; run make test-gpu P=molla-solvers T=strands_gpu RELEASE=1 A=dense_bundle_preview_tolerance"]
fn dense_bundle_preview_tolerance_matches_a_full_cholesky_step() {
    use molla_solvers::strands::{BundleDesc, CoupledConfig};
    let engine = Engine::new_gpu().unwrap();
    let mut model = BundleDesc {
        start: Vec3::new(-1.15, 1.8, 0.0),
        end: Vec3::new(1.15, 1.95, 0.0),
        strand_radius: 0.025 * (7.0_f64 / 37.0).sqrt(),
        rings: 3,
        segments: 8,
        density: 650.0,
        young_modulus: 3e7,
        shear_modulus: 1.2e7,
        tensile_strength: 2e7,
        shear_strength: 1e7,
        interface_ratio: 1.0,
        damage_rate: 0.0,
        damping: 0.35,
        pin_start: true,
    }
    .build()
    .unwrap()
    .model;
    let impulses: Vec<_> = model
        .fibers()
        .iter()
        .filter(|f| f.pin.is_none())
        .map(|f| {
            (
                f.id,
                f.center(),
                Vec3::new(0.0, -0.5, 1.0) * 1.5 * ((f.rest.position.x + 1.15) / 2.3) * f.mass.mass,
            )
        })
        .collect();
    for (id, p, j) in impulses {
        model.impulse(id, p, j).unwrap();
    }
    let mut gpu = GpuStrands::new(&engine, model).unwrap();
    let config = StepConfig {
        ground_height: None,
        coupled: Some(CoupledConfig {
            nonlinear_iterations: 1,
            linear_iterations: 512,
            relative_tolerance: 2e-5,
            absolute_tolerance: 0.0,
        }),
        ..Default::default()
    };
    for _ in 0..24 {
        gpu.step(&engine, config).unwrap();
    }
    let mut cpu = gpu.snapshot(&engine).unwrap();
    let now = std::time::Instant::now();
    step_cpu(&mut cpu, config).unwrap();
    gpu.step(&engine, config).unwrap();
    let actual = gpu.snapshot(&engine).unwrap();
    compare(&cpu, &actual, 2e-5);
    eprintln!(
        "296-cell dense Cholesky comparison: {:?}; GPU {:?}",
        now.elapsed(),
        gpu.last_coupled_diagnostics()
    );
    assert!(gpu.last_coupled_diagnostics().unwrap().relative_residual <= 2e-5);
}

#[test]
#[ignore = "requires GPU; run make test-strands-gpu"]
fn impulse_batches_are_atomic_even_after_an_earlier_valid_item() {
    let engine = Engine::new_gpu().unwrap();
    let mut gpu = GpuStrands::new(&engine, model(4, true, false)).unwrap();
    let initial = gpu.snapshot(&engine).unwrap();
    let point = initial.fiber(2).unwrap().center() + Vec3::Y * 0.02;
    let impulse = Vec3::Z * 0.03;
    assert!(
        gpu.impulses(&engine, &[(2, point, impulse), (u64::MAX, point, impulse)])
            .is_err()
    );
    compare(&initial, &gpu.snapshot(&engine).unwrap(), 1e-6);
    let items = [(2, point, impulse), (3, point, -impulse * 0.3)];
    let mut expected = initial.clone();
    for (id, point, impulse) in items {
        expected.impulse(id, point, impulse).unwrap();
    }
    gpu.impulses(&engine, &items).unwrap();
    compare(&expected, &gpu.snapshot(&engine).unwrap(), 1e-6);
}

#[test]
#[ignore = "requires GPU; run make test-strands-gpu"]
fn resident_motion_matches_the_cpu_oracle_and_preserves_pins() {
    let engine = Engine::new_gpu().unwrap();
    eprintln!("{:?}", engine.adapter_info());
    for (count, pinned) in [(1, false), (4, true), (4, false)] {
        let mut cpu = model(count, pinned, false);
        if !pinned {
            let f = cpu.fiber(0).unwrap().clone();
            cpu.set_motion(
                0,
                RigidCellMotion {
                    linear_velocity: Vec3::new(0.2, 0.1, 0.3),
                    angular_velocity: Vec3::new(0.2, -0.3, 0.4),
                    ..f.motion
                },
            )
            .unwrap();
        }
        let mut gpu = GpuStrands::new(&engine, cpu.clone()).unwrap();
        let config = StepConfig {
            ground_height: None,
            iterations: 32,
            ..StepConfig::default()
        };
        for _ in 0..120 {
            step_cpu(&mut cpu, config).unwrap();
            gpu.step(&engine, config).unwrap();
        }
        compare(&cpu, &gpu.snapshot(&engine).unwrap(), 0.0005);
    }
}

#[test]
#[ignore = "requires GPU; run make test-strands-gpu"]
fn gpu_cut_splits_at_the_plane_then_the_detached_volume_falls() {
    let engine = Engine::new_gpu().unwrap();
    let original = model(4, true, false);
    let mut gpu = GpuStrands::new(&engine, original).unwrap();
    let plane = CutPlane::new(Vec3::X * 0.371, Vec3::new(1.0, 0.0, 0.2)).unwrap();
    let before = gpu.snapshot(&engine).unwrap();
    assert!(gpu.cut(&engine, plane, &[9999], 1).is_err());
    compare(&before, &gpu.snapshot(&engine).unwrap(), 1e-10);
    let report = gpu.cut(&engine, plane, &[1], 2).unwrap();
    assert_eq!(report.replacements.len(), 1);
    let mut cpu = gpu.snapshot(&engine).unwrap();
    assert_eq!(cpu.components().len(), 2);
    let detached = report.replacements[0].1[1];
    let start = cpu.fiber(detached).unwrap().center();
    let config = StepConfig {
        ground_height: None,
        iterations: 32,
        ..StepConfig::default()
    };
    for _ in 0..120 {
        step_cpu(&mut cpu, config).unwrap();
        gpu.step(&engine, config).unwrap();
    }
    let snapshot = gpu.snapshot(&engine).unwrap();
    compare(&cpu, &snapshot, 0.0005);
    assert!(snapshot.fiber(detached).unwrap().center().y < start.y - 0.7);
}

#[test]
#[ignore = "requires GPU; run make test-strands-gpu"]
fn gpu_detects_progressive_failure_and_rejects_unrepresentable_inputs() {
    let engine = Engine::new_gpu().unwrap();
    let mut cpu = model(3, true, true);
    let mut gpu = GpuStrands::new(&engine, cpu.clone()).unwrap();
    let config = StepConfig {
        ground_height: None,
        ..StepConfig::default()
    };
    let before = gpu.snapshot(&engine).unwrap();
    assert!(
        gpu.step(
            &engine,
            StepConfig {
                dt: 1e-100,
                ..config
            }
        )
        .is_err()
    );
    compare(&before, &gpu.snapshot(&engine).unwrap(), 1e-10);
    for _ in 0..90 {
        step_cpu(&mut cpu, config).unwrap();
        gpu.step(&engine, config).unwrap();
    }
    let snapshot = gpu.snapshot(&engine).unwrap();
    assert!(snapshot.bonds().iter().any(|b| !b.intact()));
    for (a, b) in cpu.bonds().iter().zip(snapshot.bonds()) {
        assert!((a.damage - b.damage).abs() < 0.005);
    }
    compare(&cpu, &snapshot, 0.0005);
    let mut overflow = model(1, false, false);
    let f = overflow.fibers()[0].clone();
    overflow
        .set_motion(
            0,
            RigidCellMotion {
                linear_velocity: Vec3::splat(1e100),
                ..f.motion
            },
        )
        .unwrap();
    assert!(GpuStrands::new(&engine, overflow).is_err());
}

#[test]
#[ignore = "requires GPU; run make test-strands-gpu"]
fn capacity_failure_keeps_old_geometry_state_and_resources_usable() {
    let engine = Engine::new_gpu().unwrap();
    let mut gpu = GpuStrands::with_limits(
        &engine,
        model(4, true, false),
        GpuStrandLimits {
            fibers: 4,
            ..Default::default()
        },
    )
    .unwrap();
    let before = gpu.snapshot(&engine).unwrap();
    let plane = CutPlane::new(Vec3::X * 0.371, Vec3::X).unwrap();
    assert!(
        gpu.cut(&engine, plane, &[1], 42)
            .unwrap_err()
            .to_string()
            .contains("capacity")
    );
    compare(&before, &gpu.snapshot(&engine).unwrap(), 1e-10);
    gpu.step(&engine, StepConfig::default()).unwrap();
    let after = gpu.snapshot(&engine).unwrap();
    assert_eq!(after.fibers().len(), 4);
    for (a, b) in before.fibers().iter().zip(after.fibers()) {
        assert_eq!(a.shape, b.shape);
    }
}

#[test]
#[ignore = "requires GPU; run make test-strands-gpu"]
fn numerical_gpu_fault_does_not_publish_corrupted_motion() {
    let engine = Engine::new_gpu().unwrap();
    let mut gpu = GpuStrands::new(&engine, model(1, false, false)).unwrap();
    gpu.step(&engine, StepConfig::default()).unwrap();
    let before = gpu.snapshot(&engine).unwrap();
    gpu.step(
        &engine,
        StepConfig {
            dt: 1e10,
            gravity: Vec3::splat(1e38),
            iterations: 1,
            air_damping: 0.0,
            ground_height: None,
            ..StepConfig::default()
        },
    )
    .unwrap();
    assert!(gpu.snapshot(&engine).is_err());
    let (after, status) = gpu.last_valid_snapshot(&engine).unwrap();
    assert_ne!(status, 0);
    compare(&before, &after, 1e-10);
    gpu.step(&engine, StepConfig::default()).unwrap();
    let (after, again) = gpu.last_valid_snapshot(&engine).unwrap();
    assert_eq!(status, again);
    compare(&before, &after, 1e-10);
}

#[test]
#[ignore = "requires GPU; run make test-strands-gpu"]
fn gpu_cut_fragment_lands_on_its_clipped_volume_without_endpoint_spheres() {
    let engine = Engine::new_gpu().unwrap();
    let mut gpu = GpuStrands::new(&engine, model(2, true, false)).unwrap();
    let report = gpu
        .cut(
            &engine,
            CutPlane::new(Vec3::X * 0.371, Vec3::new(1.0, 0.0, 0.2)).unwrap(),
            &[1],
            11,
        )
        .unwrap();
    let id = report.replacements[0].1[1];
    let mut worst: f64 = 0.0;
    for _ in 0..90 {
        for _ in 0..8 {
            gpu.step(
                &engine,
                StepConfig {
                    iterations: 32,
                    ..Default::default()
                },
            )
            .unwrap();
        }
        let snapshot = gpu.snapshot(&engine).unwrap();
        let fiber = snapshot.fiber(id).unwrap();
        for &vertex in fiber.shape.vertices() {
            worst = worst.min(fiber.motion.pose.transform_point(vertex).y);
        }
    }
    let snapshot = gpu.snapshot(&engine).unwrap();
    let fiber = snapshot.fiber(id).unwrap();
    assert!(fiber.center().y < 0.1);
    assert!(worst > -0.002, "worst GPU clearance={worst}");
    assert!(
        fiber.motion.linear_velocity.length() < 0.1,
        "velocity={:?}",
        fiber.motion.linear_velocity
    );
}

#[test]
#[ignore = "requires GPU; run make test-strands-gpu"]
fn free_gpu_rotor_preserves_momentum_without_constraints() {
    let engine = Engine::new_gpu().unwrap();
    let mut source = model(1, false, false);
    let fiber = source.fibers()[0].clone();
    source
        .set_motion(
            0,
            RigidCellMotion {
                linear_velocity: Vec3::new(0.3, -0.2, 0.7),
                angular_velocity: Vec3::new(0.4, 1.3, -0.8),
                ..fiber.motion
            },
        )
        .unwrap();
    let mut gpu = GpuStrands::new(&engine, source).unwrap();
    let initial = gpu.snapshot(&engine).unwrap();
    let a = &initial.fibers()[0];
    let before = a.motion.momentum(a.mass).unwrap();
    for _ in 0..1000 {
        gpu.step(
            &engine,
            StepConfig {
                dt: 0.001,
                iterations: 1,
                gravity: Vec3::ZERO,
                air_damping: 0.0,
                ground_height: None,
                friction: 0.0,
                coupled: None,
                ..Default::default()
            },
        )
        .unwrap();
    }
    let final_state = gpu.snapshot(&engine).unwrap();
    let b = &final_state.fibers()[0];
    let after = b.motion.momentum(b.mass).unwrap();
    assert!(before.0.distance(after.0) < 1e-5);
    assert!(
        before.1.distance(after.1) < 5e-4,
        "before={before:?},after={after:?}"
    );
    assert!((before.2 - after.2).abs() / before.2 < 1e-3);
}

#[test]
#[ignore = "requires GPU; run make test-strands-gpu"]
fn checked_gpu_setup_catches_validation_and_other_device_use_is_rejected() {
    let engine = Engine::new_gpu().unwrap();
    let failed = engine.checked_gpu_setup(|device| {
        Ok(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("invalid strand resource fixture"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }))
    });
    assert!(
        failed
            .err()
            .unwrap()
            .to_string()
            .contains("GPU resource setup")
    );
    let mut gpu = GpuStrands::new(&engine, model(1, false, false)).unwrap();
    let other = Engine::new_gpu().unwrap();
    assert!(gpu.step(&other, StepConfig::default()).is_err());
    assert!(gpu.snapshot(&other).is_err());
    gpu.step(&engine, StepConfig::default()).unwrap();
    gpu.snapshot(&engine).unwrap();
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires GPU; run make test-strands-gpu"]
fn ordinary_steps_have_no_particle_readback_or_buffer_rebuild() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let mut gpu = GpuStrands::new(&engine, model(12, true, false)).unwrap();
    let allocations = DeviceScope::start().unwrap();
    let transfers = TransferScope::start().unwrap();
    for _ in 0..24 {
        gpu.step(
            &engine,
            StepConfig {
                ground_height: None,
                ..StepConfig::default()
            },
        )
        .unwrap();
    }
    let counts = transfers.finish();
    let allocation_counts = allocations.finish();
    assert_eq!(counts.readback_bytes, 0);
    assert_eq!(
        counts.upload_bytes,
        24 * std::mem::size_of::<molla_kernels::strands::StrandParams>() as u64
    );
    assert_eq!(allocation_counts.observations, 0);
    assert!(!counts.overflowed && !allocation_counts.invalid);
    gpu.snapshot(&engine).unwrap();
}

#[test]
#[ignore = "requires GPU; run make test-strands-gpu"]
fn curved_and_twisted_authored_rest_frames_remain_objective_on_gpu() {
    let engine = Engine::new_gpu().unwrap();
    let base = model(4, false, false);
    let mut fibers = base.fibers().to_vec();
    for (i, fiber) in fibers.iter_mut().enumerate() {
        fiber.rest.rotation = (fiber.rest.rotation
            * Quat::from_rotation_x(i as f64 * 0.17)
            * Quat::from_rotation_z(i as f64 * 0.21))
        .normalize();
        fiber.rest.position.y += 0.07 * (i as f64).sin();
        fiber.motion.pose = fiber.rest;
    }
    let mut bonds = Vec::new();
    for i in 1..fibers.len() {
        bonds.push(
            Bond::new(
                i as u64,
                [&fibers[i - 1], &fibers[i]],
                [Vec3::Z * 0.15, -Vec3::Z * 0.15],
                base.bonds()[0].material,
            )
            .unwrap(),
        );
    }
    let global = Transform::new(
        Vec3::new(0.3, -0.4, 0.2),
        Quat::from_rotation_z(0.7) * Quat::from_rotation_x(-0.4),
    );
    for fiber in &mut fibers {
        fiber.motion.pose = global.compose(&fiber.motion.pose);
    }
    let mut cpu = StrandModel::new(fibers, bonds).unwrap();
    let mut gpu = GpuStrands::new(&engine, cpu.clone()).unwrap();
    let config = StepConfig {
        gravity: Vec3::ZERO,
        air_damping: 0.0,
        ground_height: None,
        ..StepConfig::default()
    };
    for _ in 0..120 {
        step_cpu(&mut cpu, config).unwrap();
        gpu.step(&engine, config).unwrap();
    }
    compare(&cpu, &gpu.snapshot(&engine).unwrap(), 0.0001);
}

#[test]
#[ignore = "requires GPU; run make test-strands-gpu"]
fn gpu_bundle_uses_finite_interfaces_when_a_blade_crosses_old_joints() {
    let engine = Engine::new_gpu().unwrap();
    let desc = molla_solvers::strands::BundleDesc {
        start: Vec3::new(0.0, 1.5, 0.0),
        end: Vec3::new(2.0, 1.5, 0.0),
        strand_radius: 0.015,
        rings: 1,
        segments: 8,
        density: 650.0,
        young_modulus: 1e8,
        shear_modulus: 4e7,
        tensile_strength: 1e6,
        shear_strength: 5e5,
        interface_ratio: 1.0,
        damage_rate: 0.0,
        damping: 0.5,
        pin_start: true,
    };
    let bundle = desc.build().unwrap();
    let ids: Vec<_> = bundle.model.fibers().iter().map(|f| f.id).collect();
    let mut gpu = GpuStrands::new(&engine, bundle.model).unwrap();
    let plane = CutPlane::new(Vec3::new(1.137, 1.5, 0.0), Vec3::new(1.0, 0.3, -0.2)).unwrap();
    gpu.cut(&engine, plane, &ids, 71).unwrap();
    let cut = gpu.snapshot(&engine).unwrap();
    assert_eq!(cut.components().len(), 2);
    for coupled in [None, Some(Default::default())] {
        let mut cpu = cut.clone();
        let mut gpu = GpuStrands::new(&engine, cut.clone()).unwrap();
        let config = StepConfig {
            iterations: 32,
            ground_height: None,
            coupled,
            ..Default::default()
        };
        for _ in 0..12 {
            step_cpu(&mut cpu, config).unwrap();
            gpu.step(&engine, config).unwrap();
        }
        compare(&cpu, &gpu.snapshot(&engine).unwrap(), 0.0001);
    }
}

#[test]
#[ignore = "requires GPU; run make test-strands-gpu"]
fn loaded_coupled_gpu_bundle_matches_oracle_and_stays_bounded() {
    let engine = Engine::new_gpu().unwrap();
    let desc = molla_solvers::strands::BundleDesc {
        start: Vec3::new(0.0, 1.5, 0.0),
        end: Vec3::new(2.0, 1.5, 0.0),
        strand_radius: 0.015,
        rings: 1,
        segments: 8,
        density: 650.0,
        young_modulus: 1e8,
        shear_modulus: 4e7,
        tensile_strength: 1e6,
        shear_strength: 5e5,
        interface_ratio: 1.0,
        damage_rate: 0.0,
        damping: 1.0,
        pin_start: true,
    };
    let bundle = desc.build().unwrap();
    let tip = *bundle.strands[0].last().unwrap();
    let mut cpu = bundle.model.clone();
    let mut gpu = GpuStrands::new(&engine, bundle.model).unwrap();
    let mut max_gap: f64 = 0.0;
    let now = std::time::Instant::now();
    let config = StepConfig {
        iterations: 32,
        ground_height: None,
        coupled: Some(molla_solvers::strands::CoupledConfig::default()),
        ..Default::default()
    };
    for sample in 0..60 {
        for _ in 0..8 {
            if sample < 3 {
                step_cpu(&mut cpu, config).unwrap();
            }
            gpu.step(&engine, config).unwrap();
        }
        let snapshot = gpu.snapshot(&engine).unwrap();
        if sample < 3 {
            compare(&cpu, &snapshot, 0.0001);
        }
        assert!(
            gpu.last_coupled_diagnostics().unwrap().relative_residual
                <= config.coupled.unwrap().relative_tolerance as f32
        );
        assert_eq!(snapshot.components().len(), 1);
        for bond in snapshot.bonds() {
            let a = snapshot.fiber(bond.fibers[0]).unwrap();
            let b = snapshot.fiber(bond.fibers[1]).unwrap();
            let rows = molla_solvers::strands::bond_rows(bond, a, b);
            max_gap = max_gap.max(Vec3::new(rows[0].value, rows[1].value, rows[2].value).length());
        }
    }
    let snapshot = gpu.snapshot(&engine).unwrap();
    eprintln!(
        "loaded bundle: 480 steps, elapsed={:?}, tip_y={}, max_gap={max_gap}",
        now.elapsed(),
        snapshot.fiber(tip).unwrap().center().y
    );
    assert!(snapshot.fiber(tip).unwrap().center().y > 1.0);
}

#[test]
#[ignore = "requires GPU; run make test-strands-gpu"]
fn coupled_failure_is_explicit_and_never_publishes_partial_motion() {
    let engine = Engine::new_gpu().unwrap();
    let chain = model(8, true, false);
    let (fibers, mut bonds) = (chain.fibers().to_vec(), chain.bonds().to_vec());
    let material = bonds[0].material;
    bonds.push(Bond::new(8, [&fibers[2], &fibers[6]], [Vec3::ZERO; 2], material).unwrap());
    let original = StrandModel::new(fibers, bonds).unwrap();
    let mut gpu = GpuStrands::new(&engine, original.clone()).unwrap();
    let config = StepConfig {
        ground_height: None,
        coupled: Some(molla_solvers::strands::CoupledConfig {
            linear_iterations: 1,
            ..Default::default()
        }),
        ..Default::default()
    };
    gpu.step(&engine, config).unwrap();
    assert!(gpu.snapshot(&engine).is_err());
    let (retained, status) = gpu.last_valid_snapshot(&engine).unwrap();
    assert_eq!(status, 4);
    compare(&original, &retained, 1e-6);
    let stats = gpu.last_coupled_diagnostics().unwrap();
    assert_eq!(stats.iterations, 1);
    assert!(stats.relative_residual > config.coupled.unwrap().relative_tolerance as f32);
    gpu.step(&engine, StepConfig::default()).unwrap();
    let (retained, still_faulted) = gpu.last_valid_snapshot(&engine).unwrap();
    assert_eq!(still_faulted, status);
    compare(&original, &retained, 1e-6);
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires GPU; run make test-strands-gpu"]
fn coupled_cut_motion_matches_oracle_without_step_readback_or_allocations() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let original = model(8, true, false);
    let ids: Vec<_> = original.fibers().iter().map(|f| f.id).collect();
    let mut gpu = GpuStrands::new(&engine, original).unwrap();
    gpu.cut(
        &engine,
        CutPlane::new(Vec3::new(0.817, 1.5, 0.0), Vec3::new(1.0, 0.3, 0.2)).unwrap(),
        &ids,
        81,
    )
    .unwrap();
    let mut cpu = gpu.snapshot(&engine).unwrap();
    assert_eq!(cpu.components().len(), 2);
    let config = StepConfig {
        ground_height: None,
        coupled: Some(Default::default()),
        ..Default::default()
    };
    for _ in 0..24 {
        step_cpu(&mut cpu, config).unwrap();
    }
    let allocations = DeviceScope::start().unwrap();
    let transfers = TransferScope::start().unwrap();
    for _ in 0..24 {
        gpu.step(&engine, config).unwrap();
    }
    let counts = transfers.finish();
    let allocation_counts = allocations.finish();
    assert_eq!(counts.readback_bytes, 0);
    assert_eq!(allocation_counts.observations, 0);
    assert!(!counts.overflowed && !allocation_counts.invalid);
    assert_eq!(
        counts.upload_bytes,
        24 * std::mem::size_of::<molla_kernels::strands::StrandParams>() as u64
    );
    compare(&cpu, &gpu.snapshot(&engine).unwrap(), 0.0001);
}

#[test]
#[ignore = "requires a GPU"]
fn tree_preconditioned_branch_converges_in_a_few_iterations_and_matches_cpu() {
    use molla_solvers::strands::CoupledConfig;
    let engine = Engine::new_gpu().unwrap();
    let mut gpu = GpuStrands::new(&engine, model(40, true, false)).unwrap();
    let config = StepConfig {
        ground_height: None,
        coupled: Some(CoupledConfig {
            nonlinear_iterations: 1,
            linear_iterations: 512,
            relative_tolerance: 2e-5,
            absolute_tolerance: 0.0,
        }),
        ..Default::default()
    };
    for _ in 0..12 {
        gpu.step(&engine, config).unwrap();
        gpu.snapshot(&engine).unwrap();
        let diagnostics = gpu.last_coupled_diagnostics().unwrap();
        assert!(diagnostics.iterations <= 3, "{diagnostics:?}");
        assert!(diagnostics.relative_residual <= 2e-5, "{diagnostics:?}");
    }
    let mut cpu = gpu.snapshot(&engine).unwrap();
    step_cpu(&mut cpu, config).unwrap();
    gpu.step(&engine, config).unwrap();
    compare(&cpu, &gpu.snapshot(&engine).unwrap(), 2e-5);
}
