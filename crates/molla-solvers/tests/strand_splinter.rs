use molla_geometry::convex_cut::{ConvexCell, CutPlane, FaceOrigin};
use molla_math::{Quat, Transform, Vec3};
use molla_solvers::strands::{Fiber, StrandModel};

fn model() -> StrandModel {
    let mut f = Fiber::new(
        0,
        ConvexCell::prism(0.03, 0.4, 8).unwrap(),
        650.0,
        Transform::new(Vec3::Y, Quat::from_rotation_x(0.3)),
    )
    .unwrap();
    f.motion.linear_velocity = Vec3::new(0.2, 0.4, -0.1);
    f.motion.angular_velocity = Vec3::new(0.3, -0.2, 0.8);
    StrandModel::new(vec![f], vec![]).unwrap()
}

#[test]
fn damped_grab_matches_backward_euler_spring_and_damper() {
    use molla_solvers::strands::{PointGrab, StepConfig, step_cpu};
    for coupled in [None, Some(Default::default())] {
        let mut graph = model();
        let f = &graph.fibers()[0];
        let mass = f.mass.mass;
        let x = f.center();
        let v = f.motion.linear_velocity;
        let dt = 0.01;
        let grab = PointGrab {
            fiber: f.id,
            local_point: f.mass.local_com,
            target: x + Vec3::X * 0.1,
            compliance: 0.02,
            damping: 8.0,
            max_force: None,
            target_velocity: Vec3::Y * 0.2,
        };
        let expected = (v * mass
            + (grab.target - x) * (dt / grab.compliance)
            + grab.target_velocity * (dt * grab.damping))
            / (mass + dt * grab.damping + dt * dt / grab.compliance);
        step_cpu(
            &mut graph,
            StepConfig {
                dt,
                gravity: Vec3::ZERO,
                ground_height: None,
                air_damping: 0.0,
                coupled,
                grab: Some(grab),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(graph.fibers()[0].motion.linear_velocity.distance(expected) < 1e-11);
        assert!(graph.fibers()[0].center().distance(x + dt * expected) < 1e-12);
    }
}

#[test]
fn grab_force_stays_bounded_under_large_target_jumps() {
    use molla_solvers::strands::{PointGrab, StepConfig, step_cpu};
    for coupled in [None, Some(Default::default())] {
        let mut graph = model();
        let dt = 1.0 / 240.0;
        for i in 0..120 {
            let f = &graph.fibers()[0];
            let mass = f.mass.mass;
            let before = f.motion.linear_velocity;
            let grab = PointGrab {
                fiber: 0,
                local_point: f.mass.local_com,
                target: Vec3::X * if i < 60 { 100.0 } else { -100.0 },
                target_velocity: Vec3::ZERO,
                compliance: 2e-3,
                damping: 10.0,
                max_force: Some(12.0),
            };
            step_cpu(
                &mut graph,
                StepConfig {
                    dt,
                    coupled,
                    grab: Some(grab),
                    gravity: Vec3::ZERO,
                    ground_height: None,
                    air_damping: 0.0,
                    ..Default::default()
                },
            )
            .unwrap();
            let force = graph.fibers()[0].motion.linear_velocity.distance(before) * mass / dt;
            assert!(force <= 12.00001, "applied force {force}");
        }
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU"]
fn gpu_grab_caps_combined_spring_and_damping_force() {
    use molla_compute::Engine;
    use molla_solvers::strands::{GpuStrands, PointGrab, StepConfig, step_cpu};
    let engine = Engine::new_gpu().unwrap();
    for coupled in [None, Some(Default::default())] {
        let mut cpu = model();
        let mass = cpu.fibers()[0].mass.mass;
        let point = cpu.fibers()[0].mass.local_com;
        let mut gpu = GpuStrands::new(&engine, cpu.clone()).unwrap();
        let mut previous = cpu.fibers()[0].motion.linear_velocity;
        for step in 0..120 {
            let config = StepConfig {
                gravity: Vec3::ZERO,
                ground_height: None,
                air_damping: 0.0,
                coupled,
                grab: Some(PointGrab {
                    fiber: 0,
                    local_point: point,
                    target: Vec3::new(100.0, 100.0, 0.0) * if step < 60 { 1.0 } else { -1.0 },
                    target_velocity: Vec3::ZERO,
                    compliance: 2e-3,
                    damping: 10.0,
                    max_force: Some(12.0),
                }),
                ..Default::default()
            };
            step_cpu(&mut cpu, config).unwrap();
            gpu.step(&engine, config).unwrap();
            let actual = gpu.snapshot(&engine).unwrap();
            let f = &actual.fibers()[0];
            let force = mass * f.motion.linear_velocity.distance(previous) / config.dt;
            assert!(force <= 12.03, "GPU applied force {force}");
            assert!(f.center().distance(cpu.fibers()[0].center()) < 2e-4);
            previous = f.motion.linear_velocity;
        }
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU"]
fn coupled_damped_grab_matches_dense_chain_oracle_on_direction_changes() {
    use molla_compute::Engine;
    use molla_solvers::strands::{
        Bond, BondMaterial, GpuStrands, Pin, PointGrab, StepConfig, step_cpu,
    };
    let engine = Engine::new_gpu().unwrap();
    let mut fibers = Vec::new();
    for i in 0..4 {
        let mut f = Fiber::new(
            i,
            ConvexCell::prism(0.015, 0.25, 8).unwrap(),
            650.0,
            Transform::new(
                Vec3::new(i as f64 * 0.25, 1.0, 0.0),
                Quat::from_rotation_y(std::f64::consts::FRAC_PI_2),
            ),
        )
        .unwrap();
        if i == 0 {
            f.pin = Some(Pin {
                local_point: Vec3::ZERO,
                pose: f.rest,
            });
        }
        fibers.push(f);
    }
    let bonds = (1..4)
        .map(|i| {
            Bond::new(
                i as u64,
                [&fibers[i - 1], &fibers[i]],
                [Vec3::Z * 0.125, -Vec3::Z * 0.125],
                BondMaterial {
                    translation: Vec3::splat(1e-7),
                    rotation: Vec3::splat(0.02),
                    force_limit: Vec3::splat(1e8),
                    torque_limit: Vec3::splat(1e8),
                    damage_rate: 0.0,
                    damping: 1.0,
                },
            )
            .unwrap()
        })
        .collect();
    let mut cpu = StrandModel::new(fibers, bonds).unwrap();
    let mut gpu = GpuStrands::new(&engine, cpu.clone()).unwrap();
    let origin = cpu
        .fiber(3)
        .unwrap()
        .motion
        .pose
        .transform_point(Vec3::Z * 0.10);
    let dt = 1.0 / 240.0;
    for step in 0..360 {
        let t = step as f64 * dt;
        let config = StepConfig {
            dt,
            gravity: Vec3::ZERO,
            ground_height: None,
            coupled: Some(molla_solvers::strands::CoupledConfig {
                absolute_tolerance: 1e-10,
                ..Default::default()
            }),
            grab: Some(PointGrab {
                fiber: 3,
                local_point: Vec3::Z * 0.10,
                target: origin + Vec3::Y * (0.05 * (t * 6.0).sin()),
                target_velocity: Vec3::Y * (0.3 * (t * 6.0).cos()),
                compliance: 2e-4,
                damping: 30.0,
                max_force: None,
            }),
            ..Default::default()
        };
        step_cpu(&mut cpu, config).unwrap();
        gpu.step(&engine, config).unwrap();
        if step % 30 == 29 {
            let actual = gpu.snapshot(&engine).unwrap();
            for (a, b) in actual.fibers().iter().zip(cpu.fibers()) {
                assert!(
                    a.center().distance(b.center()) < 2e-4,
                    "position parity {}",
                    a.center().distance(b.center())
                );
                assert!(a.motion.linear_velocity.distance(b.motion.linear_velocity) < 0.01);
            }
        }
    }
    let tip = cpu.fiber(3).unwrap();
    assert!(tip.center().distance(origin - Vec3::X * 0.10) > 0.005);
}

#[test]
fn splinter_conserves_volume_momentum_and_creates_nonplanar_assemblies_at_hit() {
    let model = model();
    let original = &model.fibers()[0];
    let local_hit = Vec3::new(0.03, 0.0, 0.031);
    let hit = original.motion.pose.transform_point(local_hit);
    let plane = CutPlane::new(hit, original.motion.pose.rotation * Vec3::Z).unwrap();
    let edit = model.prepare_splinter(0, plane, 11).unwrap();
    let result = edit.candidate();
    assert_eq!(result.fibers().len(), 8);
    assert_eq!(result.components().len(), 2);
    assert_eq!(
        edit.report().fragments.iter().map(Vec::len).sum::<usize>(),
        8
    );
    let mass: f64 = result.fibers().iter().map(|f| f.mass.mass).sum();
    assert!((mass - original.mass.mass).abs() < 1e-12);
    let before = original.motion.momentum(original.mass).unwrap();
    let after = result
        .fibers()
        .iter()
        .map(|f| f.motion.momentum(f.mass).unwrap())
        .fold((Vec3::ZERO, Vec3::ZERO, 0.0), |a, b| {
            (a.0 + b.0, a.1 + b.1, a.2 + b.2)
        });
    assert!(before.0.distance(after.0) < 1e-12);
    assert!(before.1.distance(after.1) < 1e-12);
    assert!((before.2 - after.2).abs() < 1e-12);
    let cap_z: Vec<_> = result
        .fibers()
        .iter()
        .flat_map(|f| {
            f.shape.faces().iter().filter_map(|face| {
                if !matches!(face.origin, FaceOrigin::Fracture { id: 11, .. }) {
                    return None;
                }
                Some(
                    face.vertices
                        .iter()
                        .map(|&i| f.shape.vertices()[i as usize].z)
                        .sum::<f64>()
                        / face.vertices.len() as f64,
                )
            })
        })
        .collect();
    assert!(cap_z.iter().any(|z| (z - local_hit.z).abs() < 1e-10));
    assert!(
        cap_z.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - cap_z.iter().copied().fold(f64::INFINITY, f64::min)
            > 0.02
    );
    let mut changed = model.clone();
    let stale = changed.prepare_splinter(0, plane, 11).unwrap();
    changed.impulse(0, hit, Vec3::X).unwrap();
    assert!(changed.commit_splinter(stale).is_err());
    let mut repeated = result.clone();
    let child = &repeated.fibers()[0];
    let next = CutPlane::new(child.center(), child.motion.pose.rotation * Vec3::Z).unwrap();
    let edit = repeated.prepare_splinter(child.id, next, 12).unwrap();
    repeated.commit_splinter(edit).unwrap();
    assert_eq!(repeated.components().len(), 3);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU"]
fn gpu_splinter_is_atomic_and_grab_matches_cpu() {
    use molla_compute::Engine;
    use molla_solvers::strands::{GpuStrandLimits, GpuStrands, PointGrab, StepConfig, step_cpu};
    let engine = Engine::new_gpu().unwrap();
    let initial = model();
    let f = &initial.fibers()[0];
    let plane = CutPlane::new(f.center(), f.motion.pose.rotation * Vec3::Z).unwrap();
    let mut limited = GpuStrands::with_limits(
        &engine,
        initial.clone(),
        GpuStrandLimits {
            fibers: 2,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(limited.splinter(&engine, 0, plane, 1).is_err());
    assert_eq!(limited.snapshot(&engine).unwrap().fibers().len(), 1);
    let mut gpu = GpuStrands::new(&engine, initial.clone()).unwrap();
    let report = gpu.splinter(&engine, 0, plane, 1).unwrap();
    assert_eq!(report.fragments.iter().map(Vec::len).sum::<usize>(), 8);
    let snapshot = gpu.snapshot(&engine).unwrap();
    assert_eq!(snapshot.components().len(), 2);
    let mut oracle = snapshot;
    let config = StepConfig {
        ground_height: None,
        gravity: Vec3::ZERO,
        coupled: Some(Default::default()),
        ..Default::default()
    };
    for _ in 0..8 {
        step_cpu(&mut oracle, config).unwrap();
        gpu.step(&engine, config).unwrap();
    }
    let result = gpu.snapshot(&engine).unwrap();
    for (a, b) in result.fibers().iter().zip(oracle.fibers()) {
        assert!(a.center().distance(b.center()) < 2e-5);
    }
    let mut cpu = initial.clone();
    let mut gpu = GpuStrands::new(&engine, initial).unwrap();
    for i in 0..60 {
        let config = StepConfig {
            gravity: Vec3::ZERO,
            ground_height: None,
            grab: Some(PointGrab {
                fiber: 0,
                local_point: Vec3::Z * 0.15,
                target: Vec3::new(i as f64 * 0.002, 1.1, 0.1),
                compliance: 1e-4,
                target_velocity: Vec3::ZERO,
                damping: 0.0,
                max_force: None,
            }),
            ..Default::default()
        };
        step_cpu(&mut cpu, config).unwrap();
        gpu.step(&engine, config).unwrap();
    }
    let singleton = StepConfig {
        gravity: Vec3::ZERO,
        ground_height: None,
        coupled: Some(molla_solvers::strands::CoupledConfig {
            absolute_tolerance: 1e-10,
            ..Default::default()
        }),
        grab: Some(PointGrab {
            fiber: 0,
            local_point: Vec3::Z * 0.15,
            target: Vec3::new(0.12, 1.1, 0.1),
            target_velocity: Vec3::X * 0.2,
            compliance: 1e-4,
            damping: 10.0,
            max_force: Some(12.0),
        }),
        ..Default::default()
    };
    step_cpu(&mut cpu, singleton).unwrap();
    gpu.step(&engine, singleton).unwrap();
    let actual = gpu.snapshot(&engine).unwrap();
    assert!(
        actual.fibers()[0]
            .center()
            .distance(cpu.fibers()[0].center())
            < 2e-5
    );
    assert!(
        actual.fibers()[0]
            .motion
            .angular_velocity
            .distance(cpu.fibers()[0].motion.angular_velocity)
            < 0.005
    );
    let invalid = StepConfig {
        grab: Some(PointGrab {
            fiber: 999,
            local_point: Vec3::ZERO,
            target: Vec3::ZERO,
            compliance: 1e-4,
            target_velocity: Vec3::ZERO,
            damping: 0.0,
            max_force: None,
        }),
        ..Default::default()
    };
    assert!(gpu.step(&engine, invalid).is_err());
    let retained = gpu.snapshot(&engine).unwrap();
    assert_eq!(
        actual.fibers()[0].motion.pose,
        retained.fibers()[0].motion.pose
    );
    assert_eq!(
        actual.fibers()[0].motion.linear_velocity,
        retained.fibers()[0].motion.linear_velocity
    );
    assert_eq!(
        actual.fibers()[0].motion.angular_velocity,
        retained.fibers()[0].motion.angular_velocity
    );
}
