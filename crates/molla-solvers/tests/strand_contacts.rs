use molla_geometry::{
    cell_contact::cell_manifold,
    convex_cut::{ConvexCell, CutPlane},
};
use molla_math::{Quat, Transform, Vec3};
use molla_solvers::strands::{Bond, BondMaterial, Fiber, Pin, StepConfig, StrandModel, step_cpu};

fn impact() -> StrandModel {
    let mut a = Fiber::new(
        0,
        ConvexCell::prism(0.08, 0.4, 6).unwrap(),
        650.0,
        Transform::new(-Vec3::Z * 0.202, Quat::IDENTITY),
    )
    .unwrap();
    let mut b = Fiber::new(
        1,
        a.shape.clone(),
        650.0,
        Transform::new(Vec3::Z * 0.202, Quat::IDENTITY),
    )
    .unwrap();
    a.motion.linear_velocity = Vec3::Z;
    b.motion.linear_velocity = -Vec3::Z;
    StrandModel::new(vec![a, b], vec![]).unwrap()
}

fn config() -> StepConfig {
    StepConfig {
        gravity: Vec3::ZERO,
        ground_height: None,
        air_damping: 0.0,
        contact_margin: Some(1e-5),
        pair_restitution: 0.4,
        friction: 0.3,
        ..Default::default()
    }
}

fn junction(broken: bool) -> StrandModel {
    let shape = ConvexCell::prism(0.03, 0.3, 8).unwrap();
    let mut fibers = vec![Fiber::new(0, shape.clone(), 650.0, Transform::IDENTITY).unwrap()];
    fibers[0].pin = Some(Pin {
        local_point: Vec3::ZERO,
        pose: Transform::IDENTITY,
    });
    for id in 1..=2 {
        fibers.push(
            Fiber::new(
                id,
                shape.clone(),
                650.0,
                Transform::new(Vec3::Z * 0.15, Quat::IDENTITY),
            )
            .unwrap(),
        );
    }
    let material = BondMaterial {
        translation: Vec3::splat(1e-6),
        rotation: Vec3::splat(1e-3),
        force_limit: Vec3::splat(1e6),
        torque_limit: Vec3::splat(1e6),
        damage_rate: 0.0,
        damping: 0.5,
    };
    let mut bonds = (1..=2)
        .map(|i| {
            Bond::new(
                i as u64,
                [&fibers[0], &fibers[i]],
                [Vec3::Z * 0.15, Vec3::ZERO],
                material,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    if broken {
        bonds[1].damage = 1.0;
    }
    StrandModel::new(fibers, bonds).unwrap()
}

#[test]
fn junction_neighbors_do_not_fight_their_attachments_but_detached_cells_collide() {
    let config = StepConfig {
        contact_exclusion_hops: 2,
        ..config()
    };
    let mut intact = junction(false);
    for _ in 0..8 {
        step_cpu(&mut intact, config).unwrap();
    }
    assert!(intact.fiber(1).unwrap().center().distance(Vec3::Z * 0.15) < 1e-10);
    assert!(intact.fiber(2).unwrap().center().distance(Vec3::Z * 0.15) < 1e-10);
    let mut broken = junction(true);
    for _ in 0..32 {
        step_cpu(&mut broken, config).unwrap();
    }
    assert!(
        broken
            .fiber(1)
            .unwrap()
            .center()
            .distance(broken.fiber(2).unwrap().center())
            > 0.02
    );
    assert_eq!(broken.components().len(), 2);
    assert!(
        StepConfig {
            contact_exclusion_hops: 3,
            ..config
        }
        .validate()
        .is_err()
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; run make test-strands-gpu"]
fn gpu_junction_exclusions_match_cpu_and_reenable_contact_after_detachment() {
    use molla_solvers::strands::{GpuStrandLimits, GpuStrands};
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let config = StepConfig {
        contact_exclusion_hops: 2,
        ..config()
    };
    for broken in [false, true] {
        let mut cpu = junction(broken);
        let mut gpu = GpuStrands::with_limits(
            &engine,
            cpu.clone(),
            GpuStrandLimits {
                contact_pairs: 16,
                ..Default::default()
            },
        )
        .unwrap();
        for _ in 0..32 {
            step_cpu(&mut cpu, config).unwrap();
            gpu.step(&engine, config).unwrap();
        }
        let actual = gpu.snapshot(&engine).unwrap();
        for (a, b) in actual.fibers().iter().zip(cpu.fibers()) {
            assert!(a.center().distance(b.center()) < 0.003);
        }
        let separation = actual
            .fiber(1)
            .unwrap()
            .center()
            .distance(actual.fiber(2).unwrap().center());
        if broken {
            assert!(separation > 0.02);
        } else {
            assert!(separation < 1e-6);
        }
    }
}

fn penetration(model: &StrandModel) -> f64 {
    let a = &model.fibers()[0];
    let b = &model.fibers()[1];
    cell_manifold(&a.shape, a.motion.pose, &b.shape, b.motion.pose, 0.0)
        .unwrap()
        .map_or(0.0, |c| {
            c.points
                .iter()
                .map(|p| (p.a - p.b).dot(c.normal))
                .fold(0.0, f64::max)
        })
}

fn totals(model: &StrandModel) -> (Vec3, f64) {
    model
        .fibers()
        .iter()
        .fold((Vec3::ZERO, 0.0), |(momentum, energy), f| {
            let (p, _, k) = f.motion.momentum(f.mass).unwrap();
            (momentum + p, energy + k)
        })
}

fn obstacle() -> StrandModel {
    let rotation = Quat::from_rotation_x(-std::f64::consts::FRAC_PI_2);
    let mut fixed = Fiber::new(
        0,
        ConvexCell::prism(0.3, 0.2, 6).unwrap(),
        650.0,
        Transform::new(Vec3::ZERO, rotation),
    )
    .unwrap();
    fixed.pin = Some(Pin {
        local_point: Vec3::ZERO,
        pose: fixed.rest,
    });
    let falling = Fiber::new(
        1,
        ConvexCell::prism(0.08, 0.2, 6).unwrap(),
        650.0,
        Transform::new(Vec3::Y * 0.6, rotation),
    )
    .unwrap();
    StrandModel::new(vec![fixed, falling], vec![]).unwrap()
}

#[test]
fn cell_pairs_rebound_without_adding_energy_or_linear_momentum() {
    let mut model = impact();
    let before = totals(&model);
    for _ in 0..12 {
        step_cpu(&mut model, config()).unwrap();
    }
    assert!(penetration(&model) < 1e-6);
    assert!(model.fibers()[0].motion.linear_velocity.z < -0.05);
    assert!(model.fibers()[1].motion.linear_velocity.z > 0.05);
    assert!(totals(&model).0.distance(before.0) < 1e-10);
    assert!(totals(&model).1 <= before.1 * 1.001);
}

#[test]
fn fixed_convex_obstacle_supports_a_falling_cell_without_a_floor() {
    let mut model = obstacle();
    let fixed_center = model.fibers()[0].center();
    let config = StepConfig {
        gravity: -Vec3::Y * 9.81,
        ..config()
    };
    let mut bounced = false;
    for _ in 0..240 {
        step_cpu(&mut model, config).unwrap();
        assert!(penetration(&model) < 1e-4);
        bounced |= model.fibers()[1].motion.linear_velocity.y > 0.1;
    }
    assert!(bounced);
    assert!(model.fibers()[1].center().y > 0.199);
    assert_eq!(model.fibers()[0].center(), fixed_center);
}

#[test]
fn broken_direct_bond_reenables_surface_contact() {
    let source = impact();
    let mut bodies = source.fibers().to_vec();
    bodies[0].motion.pose.position.z = -0.198;
    bodies[1].motion.pose.position.z = 0.198;
    for f in &mut bodies {
        f.motion.linear_velocity = Vec3::ZERO;
        f.rest = f.motion.pose;
    }
    let material = BondMaterial {
        translation: Vec3::splat(1.0),
        rotation: Vec3::splat(1.0),
        force_limit: Vec3::splat(1e6),
        torque_limit: Vec3::splat(1e6),
        damage_rate: 0.0,
        damping: 0.0,
    };
    let bond = Bond::new(0, [&bodies[0], &bodies[1]], [Vec3::ZERO; 2], material).unwrap();
    let mut intact = StrandModel::new(bodies.clone(), vec![bond.clone()]).unwrap();
    let mut broken = StrandModel::new(
        bodies,
        vec![Bond {
            damage: 1.0,
            ..bond
        }],
    )
    .unwrap();
    step_cpu(&mut intact, config()).unwrap();
    step_cpu(&mut broken, config()).unwrap();
    assert!(penetration(&intact) > 0.0039);
    assert!(penetration(&broken) < 1e-6);
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use molla_compute::Engine;
    use molla_solvers::strands::{GpuStrandLimits, GpuStrands};

    fn solver(engine: &Engine, model: StrandModel, capacity: usize) -> GpuStrands {
        GpuStrands::with_limits(
            engine,
            model,
            GpuStrandLimits {
                contact_pairs: capacity,
                ..Default::default()
            },
        )
        .unwrap()
    }

    #[test]
    #[ignore = "requires GPU; run make test-strands-gpu"]
    fn sparse_contact_masks_preserve_high_index_impacts_and_clear_empty_frames() {
        let engine = Engine::new_gpu().unwrap();
        let mut fibers = Vec::new();
        for id in 0..69 {
            let rest = Transform::new(Vec3::new(5.0 + id as f64, 0.0, 0.0), Quat::IDENTITY);
            let mut f =
                Fiber::new(id, ConvexCell::prism(0.04, 0.2, 6).unwrap(), 650.0, rest).unwrap();
            f.pin = Some(Pin {
                local_point: Vec3::ZERO,
                pose: rest,
            });
            fibers.push(f);
        }
        for mut f in impact().fibers().to_vec() {
            f.id += 69;
            fibers.push(f);
        }
        let mut cpu = StrandModel::new(fibers, vec![]).unwrap();
        let mut gpu = solver(&engine, cpu.clone(), 2485);
        for _ in 0..24 {
            step_cpu(&mut cpu, config()).unwrap();
            gpu.step(&engine, config()).unwrap();
            let actual = gpu.snapshot(&engine).unwrap();
            for id in 69..71 {
                let a = actual.fiber(id).unwrap();
                let b = cpu.fiber(id).unwrap();
                assert!(a.center().distance(b.center()) < 3e-4);
                assert!(a.motion.linear_velocity.distance(b.motion.linear_velocity) < 3e-3);
            }
        }
        assert!(cpu.fiber(69).unwrap().motion.linear_velocity.z < 0.0);
    }

    #[test]
    #[ignore = "requires GPU; run make test-strands-gpu"]
    fn gpu_surface_impacts_and_obstacles_match_cpu() {
        let engine = Engine::new_gpu().unwrap();
        for (mut cpu, config, steps) in [
            (impact(), config(), 12),
            (
                obstacle(),
                StepConfig {
                    gravity: -Vec3::Y * 9.81,
                    ..config()
                },
                240,
            ),
        ] {
            let mut gpu = solver(&engine, cpu.clone(), 1);
            for step in 0..steps {
                step_cpu(&mut cpu, config).unwrap();
                gpu.step(&engine, config).unwrap();
                let snapshot = gpu.snapshot(&engine).unwrap();
                assert!(penetration(&snapshot) < 2e-4, "step {step}");
                for (a, b) in cpu.fibers().iter().zip(snapshot.fibers()) {
                    assert!(
                        a.center().distance(b.center()) < 0.003,
                        "step {step}, CPU {:?}, GPU {:?}",
                        a.center(),
                        b.center()
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "requires GPU; run make test-strands-gpu"]
    fn gpu_contact_capacity_rejects_cut_without_changing_the_old_model() {
        let engine = Engine::new_gpu().unwrap();
        let mut gpu = solver(&engine, impact(), 1);
        let before = gpu.snapshot(&engine).unwrap();
        let plane = CutPlane::new(-Vec3::Z * 0.2, Vec3::Z).unwrap();
        assert!(
            gpu.cut(&engine, plane, &[0], 1)
                .unwrap_err()
                .to_string()
                .contains("contact pair capacity")
        );
        let after = gpu.snapshot(&engine).unwrap();
        assert_eq!(after.fibers().len(), before.fibers().len());
        for (a, b) in after.fibers().iter().zip(before.fibers()) {
            assert_eq!(a.center(), b.center());
        }
        gpu.step(&engine, config()).unwrap();
        gpu.snapshot(&engine).unwrap();
        let mut disabled = GpuStrands::new(&engine, impact()).unwrap();
        assert!(disabled.step(&engine, config()).is_err());
    }
}
