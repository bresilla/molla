use molla_geometry::convex_cut::{ConvexCell, CutPlane, RigidCellMotion};
use molla_math::{Quat, Transform, Vec3};
use molla_sim::MassProperties;
use molla_solvers::strands::{
    Bond, BondMaterial, Fiber, Pin, StepConfig, StrandModel, bond_rows, step_cpu,
};

fn material() -> BondMaterial {
    BondMaterial {
        translation: Vec3::splat(1e-7),
        rotation: Vec3::splat(0.003),
        force_limit: Vec3::splat(1e6),
        torque_limit: Vec3::splat(1e6),
        damage_rate: 0.0,
        damping: 0.5,
    }
}

fn chain(count: usize, pinned: bool, curved: bool) -> StrandModel {
    let points: Vec<_> = (0..=count)
        .map(|i| {
            let t = i as f64 / count as f64;
            Vec3::new(
                2.0 * t,
                2.0 + if curved { 0.3 * (t * 2.4).sin() } else { 0.0 },
                0.0,
            )
        })
        .collect();
    let mut fibers = Vec::new();
    for i in 0..count {
        let d = points[i + 1] - points[i];
        let rest = Transform::new(
            (points[i + 1] + points[i]) * 0.5,
            Quat::from_rotation_arc(Vec3::Z, d.normalize()),
        );
        let mut fiber = Fiber::new(
            i as u64,
            ConvexCell::prism(0.04, d.length(), 8).unwrap(),
            650.0,
            rest,
        )
        .unwrap();
        if i == 0 && pinned {
            fiber.pin = Some(Pin {
                local_point: -Vec3::Z * d.length() * 0.5,
                pose: rest,
            });
        }
        fibers.push(fiber);
    }
    let mut bonds = Vec::new();
    for i in 1..count {
        let anchors = [
            fibers[i - 1].rest.inverse_transform_point(points[i]),
            fibers[i].rest.inverse_transform_point(points[i]),
        ];
        bonds.push(Bond::new(i as u64, [&fibers[i - 1], &fibers[i]], anchors, material()).unwrap());
    }
    StrandModel::new(fibers, bonds).unwrap()
}

fn free_config() -> StepConfig {
    StepConfig {
        gravity: Vec3::ZERO,
        ground_height: None,
        air_damping: 0.0,
        ..StepConfig::default()
    }
}

fn totals(model: &StrandModel) -> (f64, Vec3, Vec3, f64) {
    let mut total = (0.0, Vec3::ZERO, Vec3::ZERO, 0.0);
    for fiber in model.fibers() {
        let (p, l, k) = fiber.motion.momentum(fiber.mass).unwrap();
        total.0 += fiber.mass.mass;
        total.1 += p;
        total.2 += l;
        total.3 += k;
    }
    total
}

#[test]
fn rotation_rows_resolve_sub_microradian_twist_continuously() {
    let graph = chain(2, false, false);
    let a = graph.fibers()[0].clone();
    let original = &graph.fibers()[1];
    for angle in [1e-12, 1e-10, 1e-8, 1e-6, -1e-8] {
        let mut b = original.clone();
        b.motion.pose.rotation = a.motion.pose.rotation * Quat::from_rotation_z(angle);
        let rows = bond_rows(&graph.bonds()[0], &a, &b);
        assert!((rows[5].value - angle).abs() < 1e-14);
        assert!(rows[3].value.abs() + rows[4].value.abs() < 1e-14);
    }
}

#[test]
fn coupled_axial_chain_matches_independent_tridiagonal_implicit_solve() {
    let base = chain(12, true, false);
    let mut bonds = base.bonds().to_vec();
    for bond in &mut bonds {
        bond.material.translation = Vec3::splat(1e-9);
        bond.material.damping = 0.0;
    }
    let mut model = StrandModel::new(base.fibers().to_vec(), bonds).unwrap();
    let initial: Vec<_> = model.fibers().iter().map(|f| f.center()).collect();
    let config = StepConfig {
        gravity: Vec3::X * 100.0,
        coupled: Some(molla_solvers::strands::CoupledConfig {
            nonlinear_iterations: 1,
            ..Default::default()
        }),
        ..free_config()
    };
    let n = model.fibers().len() - 1;
    let stiffness = config.dt * config.dt / 1e-9;
    let mut diagonal = vec![0.0; n];
    let mut load = vec![0.0; n];
    for i in 0..n {
        let mass = model.fibers()[i + 1].mass.mass;
        diagonal[i] = mass
            + if i + 1 == n {
                stiffness
            } else {
                2.0 * stiffness
            };
        load[i] = mass * config.gravity.x * config.dt * config.dt;
        if i > 0 {
            let factor = -stiffness / diagonal[i - 1];
            diagonal[i] += factor * stiffness;
            load[i] -= factor * load[i - 1];
        }
    }
    for i in (0..n).rev() {
        if i + 1 < n {
            load[i] += stiffness * load[i + 1];
        }
        load[i] /= diagonal[i];
    }
    step_cpu(&mut model, config).unwrap();
    for (i, fiber) in model.fibers().iter().enumerate().skip(1) {
        let displacement = fiber.center() - initial[i];
        assert!((displacement.x - load[i - 1]).abs() < 1e-12);
        assert!(displacement.y.abs() + displacement.z.abs() < 1e-12);
    }
}

#[test]
fn exact_constraint_jacobians_match_finite_differences() {
    let graph = chain(2, false, true);
    let mut a = graph.fibers()[0].clone();
    let mut b = graph.fibers()[1].clone();
    a.motion.pose.rotation = (Quat::from_rotation_x(0.27) * a.motion.pose.rotation).normalize();
    b.motion.pose.rotation = (Quat::from_rotation_y(-0.43) * b.motion.pose.rotation).normalize();
    b.motion.pose.position += Vec3::new(0.02, -0.07, 0.04);
    let bond = &graph.bonds()[0];
    let rows = bond_rows(bond, &a, &b);
    let eps = 1e-6;
    for side in 0..2 {
        for dof in 0..6 {
            let mut differences = [[0.0; 6]; 2];
            for (sign_index, sign) in [-1.0, 1.0].into_iter().enumerate() {
                let mut pair = [a.clone(), b.clone()];
                let fiber = &mut pair[side];
                let axis = [Vec3::X, Vec3::Y, Vec3::Z][dof % 3];
                if dof < 3 {
                    fiber.motion.pose.position += axis * (sign * eps);
                } else {
                    let center = fiber.center();
                    fiber.motion.pose.rotation = (Quat::from_scaled_axis(axis * (sign * eps))
                        * fiber.motion.pose.rotation)
                        .normalize();
                    fiber.motion.pose.position =
                        center - fiber.motion.pose.rotation * fiber.mass.local_com;
                }
                differences[sign_index] = bond_rows(bond, &pair[0], &pair[1]).map(|r| r.value);
            }
            for row in 0..6 {
                let numeric = (differences[1][row] - differences[0][row]) / (2.0 * eps);
                let analytic = if dof < 3 {
                    rows[row].linear[side][dof]
                } else {
                    rows[row].angular[side][dof - 3]
                };
                assert!(
                    (numeric - analytic).abs() < 2e-9,
                    "row={row},side={side},dof={dof}, numeric={numeric},analytic={analytic}"
                );
            }
        }
    }
}

#[test]
fn curved_material_rest_is_objective_not_a_world_space_attractor() {
    let mut graph = chain(5, false, true);
    let rigid = Transform::new(
        Vec3::new(3.0, -2.0, 1.0),
        Quat::from_rotation_y(0.73) * Quat::from_rotation_z(-0.29),
    );
    for original in graph.fibers().to_vec() {
        let motion = RigidCellMotion {
            pose: rigid.compose(&original.motion.pose),
            ..original.motion
        };
        graph.set_motion(original.id, motion).unwrap();
    }
    let before: Vec<_> = graph.fibers().iter().map(|f| f.motion.pose).collect();
    let report = step_cpu(&mut graph, free_config()).unwrap();
    assert!(report.elastic_energy < 1e-20);
    for (fiber, pose) in graph.fibers().iter().zip(before) {
        assert!(fiber.motion.pose.position.distance(pose.position) < 1e-12);
        assert!(fiber.motion.pose.rotation.dot(pose.rotation).abs() > 1.0 - 1e-12);
    }
}

#[test]
fn cuts_preserve_motion_rest_bonds_and_closed_geometry_without_snapping() {
    let mut graph = chain(4, false, true);
    let omega = Vec3::new(0.3, -0.8, 0.7);
    for fiber in graph.fibers().to_vec() {
        graph
            .set_motion(
                fiber.id,
                RigidCellMotion {
                    linear_velocity: Vec3::new(0.1, 0.2, -0.3) + omega.cross(fiber.center()),
                    angular_velocity: omega,
                    ..fiber.motion
                },
            )
            .unwrap();
    }
    let before = totals(&graph);
    let revision = graph.revision();
    let parent = graph.fiber(2).unwrap().clone();
    let plane = CutPlane::new(Vec3::new(1.371, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.2)).unwrap();
    let transaction = graph.prepare_cut(plane, &[2], 42).unwrap();
    assert_eq!(graph.revision(), revision);
    assert_eq!(graph.fibers().len(), 4);
    assert_eq!(transaction.report().replacements.len(), 1);
    let ids = transaction.report().replacements[0].1;
    assert_eq!(transaction.candidate().components().len(), 2);
    graph.commit_cut(transaction).unwrap();
    let after = totals(&graph);
    assert!((after.0 - before.0).abs() < 1e-12);
    assert!(after.1.distance(before.1) < 1e-12);
    assert!(after.2.distance(before.2) < 1e-11);
    assert!((after.3 - before.3).abs() < 1e-11);
    assert_eq!(graph.bonds().len(), 3);
    for id in ids {
        let child = graph.fiber(id).unwrap();
        assert_eq!(child.rest, parent.rest);
        assert_eq!(child.motion.pose, parent.motion.pose);
        child.shape.validate().unwrap();
    }
    assert!(graph.fiber(2).is_none());
    let second = graph
        .prepare_cut(CutPlane::new(Vec3::X * 1.63, Vec3::X).unwrap(), &[3], 43)
        .unwrap();
    graph.commit_cut(second).unwrap();
    assert_eq!(graph.components().len(), 3);
}

#[test]
fn cut_at_an_existing_interface_severs_the_bond_without_tiny_fragments() {
    let mut graph = chain(4, true, false);
    let transaction = graph
        .prepare_cut(CutPlane::new(Vec3::X * 1.0, Vec3::X).unwrap(), &[1, 2], 5)
        .unwrap();
    assert!(transaction.report().replacements.is_empty());
    assert_eq!(transaction.report().severed_bonds, vec![2]);
    graph.commit_cut(transaction).unwrap();
    assert_eq!(graph.components(), vec![vec![0, 1], vec![2, 3]]);
}

#[test]
fn root_pinning_stays_only_on_the_attached_child_and_errors_are_atomic() {
    let mut graph = chain(2, true, false);
    let plane = CutPlane::new(Vec3::X * 0.371, Vec3::X).unwrap();
    let transaction = graph.prepare_cut(plane, &[0], 7).unwrap();
    let ids = transaction.report().replacements[0].1;
    graph.commit_cut(transaction).unwrap();
    assert!(graph.fiber(ids[0]).unwrap().pin.is_some());
    assert!(graph.fiber(ids[1]).unwrap().pin.is_none());
    let revision = graph.revision();
    let shapes: Vec<_> = graph.fibers().iter().map(|f| f.shape.clone()).collect();
    assert!(graph.prepare_cut(plane, &[999], 8).is_err());
    assert!(graph.prepare_cut(plane, &[ids[0], ids[0]], 8).is_err());
    assert!(
        graph
            .prepare_cut(
                CutPlane::new(Vec3::X * (0.371 - 1e-12), Vec3::X).unwrap(),
                &[ids[0]],
                8
            )
            .is_err()
    );
    assert_eq!(graph.revision(), revision);
    assert_eq!(
        graph
            .fibers()
            .iter()
            .map(|f| f.shape.clone())
            .collect::<Vec<_>>(),
        shapes
    );
    let stale = graph
        .prepare_cut(CutPlane::new(Vec3::X * 1.7, Vec3::X).unwrap(), &[1], 9)
        .unwrap();
    graph.impulse(1, Vec3::X, Vec3::Y).unwrap();
    assert!(graph.commit_cut(stale).is_err());
    let transaction = graph.prepare_cut(plane, &[], 9).unwrap();
    let mut unrelated = chain(2, true, false);
    assert!(unrelated.commit_cut(transaction).is_err());
    let mut fork = graph.clone();
    graph.impulse(1, Vec3::ZERO, Vec3::X).unwrap();
    fork.impulse(1, Vec3::ZERO, -Vec3::X).unwrap();
    let transaction = graph.prepare_cut(plane, &[], 10).unwrap();
    assert!(fork.commit_cut(transaction).is_err());
}

#[test]
fn rotating_free_fiber_conserves_linear_and_angular_momentum() {
    let mut graph = chain(1, false, false);
    let fiber = graph.fibers()[0].clone();
    graph
        .set_motion(
            0,
            RigidCellMotion {
                linear_velocity: Vec3::new(0.3, -0.2, 0.7),
                angular_velocity: Vec3::new(0.4, 1.3, -0.8),
                ..fiber.motion
            },
        )
        .unwrap();
    let before = totals(&graph);
    let config = StepConfig {
        dt: 1.0 / 1000.0,
        ..free_config()
    };
    for _ in 0..1000 {
        step_cpu(&mut graph, config).unwrap();
    }
    let after = totals(&graph);
    assert!(before.1.distance(after.1) < 1e-10);
    assert!(
        before.2.distance(after.2) < 1e-9,
        "before={before:?},after={after:?}"
    );
    assert!((before.3 - after.3).abs() / before.3 < 1e-5);
}

#[test]
fn bent_branch_oscillates_and_recovers_authored_shape_without_gravity() {
    let mut graph = chain(3, true, true);
    let tip = graph.fiber(2).unwrap().clone();
    let mut displaced = tip.motion;
    displaced.pose.position.z += 0.12;
    graph.set_motion(2, displaced).unwrap();
    let root = graph.fiber(0).unwrap().motion.pose;
    let mut signs = [false; 2];
    let mut last_energy = 0.0;
    let config = StepConfig {
        iterations: 48,
        air_damping: 0.7,
        ..free_config()
    };
    for _ in 0..1200 {
        let report = step_cpu(&mut graph, config).unwrap();
        let z = graph.fiber(2).unwrap().motion.pose.position.z;
        if z > 1e-3 {
            signs[0] = true;
        }
        if z < -1e-3 {
            signs[1] = true;
        }
        last_energy = report.elastic_energy;
    }
    assert_eq!(graph.fiber(0).unwrap().motion.pose, root);
    assert_eq!(signs, [true, true]);
    let error = graph
        .fiber(2)
        .unwrap()
        .motion
        .pose
        .position
        .distance(tip.rest.position);
    assert!(
        error < 0.02,
        "tip error={error},elastic energy={last_energy}"
    );
}

#[test]
fn invalid_steps_leave_the_model_and_revision_unchanged() {
    let mut graph = chain(3, true, false);
    let before = graph.revision();
    let total = totals(&graph);
    for dt in [0.0, -0.1, f64::NAN, f64::INFINITY, 1e-300] {
        assert!(
            step_cpu(
                &mut graph,
                StepConfig {
                    dt,
                    ..free_config()
                }
            )
            .is_err()
        );
    }
    assert_eq!(graph.revision(), before);
    assert_eq!(totals(&graph), total);
    assert!(graph.impulse(2, Vec3::ZERO, Vec3::splat(f64::NAN)).is_err());
    assert_eq!(graph.revision(), before);
}

#[test]
fn bond_failure_is_load_driven_and_can_leave_a_partial_attachment() {
    let base = chain(3, true, false);
    let mut bonds = base.bonds().to_vec();
    let mut duplicate = bonds[0].clone();
    duplicate.id = 99;
    duplicate.material.force_limit = Vec3::splat(0.01);
    duplicate.material.torque_limit = Vec3::splat(0.01);
    duplicate.material.damage_rate = 0.05;
    bonds.push(duplicate);
    let mut graph = StrandModel::new(base.fibers().to_vec(), bonds).unwrap();
    let config = StepConfig {
        ground_height: None,
        iterations: 48,
        ..StepConfig::default()
    };
    let mut observed_partial_damage = false;
    let mut broken = false;
    for step in 0..200 {
        if step == 30 {
            graph
                .impulse(1, graph.fiber(1).unwrap().center(), Vec3::Y * -5.0)
                .unwrap();
        }
        let report = step_cpu(&mut graph, config).unwrap();
        let damage = graph.bonds().iter().find(|b| b.id == 99).unwrap().damage;
        observed_partial_damage |= damage > 0.0 && damage < 1.0;
        broken |= report.broken_bonds.contains(&99);
        if broken {
            break;
        }
    }
    assert!(observed_partial_damage);
    assert!(broken);
    assert_eq!(graph.components().len(), 1);
    assert!(graph.bonds()[0].intact());
}

#[test]
fn child_mass_properties_are_additive_in_the_authored_frame() {
    let graph = chain(1, false, false);
    let plane = CutPlane::new(Vec3::new(0.713, 2.0, 0.0), Vec3::new(1.0, 0.3, 0.2)).unwrap();
    let cut = graph.prepare_cut(plane, &[0], 7).unwrap();
    let total = cut
        .candidate()
        .fibers()
        .iter()
        .fold(MassProperties::ZERO, |sum, f| sum.combined(f.mass).unwrap());
    let original = graph.fibers()[0].mass;
    assert!((total.mass - original.mass).abs() < 1e-12);
    assert!(total.local_com.distance(original.local_com) < 1e-12);
    assert!(
        (total.inertia - original.inertia)
            .to_cols_array()
            .iter()
            .all(|x| x.abs() < 1e-12)
    );
}

#[test]
fn twist_near_half_turn_has_a_nonzero_restoring_gradient() {
    let graph = chain(2, true, false);
    let a = &graph.fibers()[0];
    let mut b = graph.fibers()[1].clone();
    b.motion.pose.rotation =
        (b.motion.pose.rotation * Quat::from_rotation_z(std::f64::consts::PI - 1e-5)).normalize();
    let rows = bond_rows(&graph.bonds()[0], a, &b);
    assert!((rows[5].value - (std::f64::consts::PI - 1e-5)).abs() < 1e-10);
    assert!(rows[5].angular[1].length() > 0.99);
    let mut opposite = b.clone();
    opposite.motion.pose.rotation = -opposite.motion.pose.rotation;
    let equivalent = bond_rows(&graph.bonds()[0], a, &opposite);
    for i in 0..6 {
        assert!((rows[i].value - equivalent[i].value).abs() < 1e-12);
    }
}

#[test]
fn compliant_translation_converges_to_the_analytic_oscillator_with_timestep() {
    let base = chain(2, true, false);
    let mut bond = Bond::new(
        0,
        [&base.fibers()[0], &base.fibers()[1]],
        [Vec3::ZERO; 2],
        material(),
    )
    .unwrap();
    bond.material.translation = Vec3::splat(0.02);
    bond.material.damping = 0.0;
    let frequency = (1.0 / (base.fibers()[1].mass.mass * 0.02)).sqrt();
    let duration = 0.4;
    let expected = 0.1 * (frequency * duration).cos();
    let mut errors = Vec::new();
    for steps in [48, 96, 192] {
        let mut graph = StrandModel::new(base.fibers().to_vec(), vec![bond.clone()]).unwrap();
        let mut motion = graph.fiber(1).unwrap().motion;
        motion.pose.position.x += 0.1;
        graph.set_motion(1, motion).unwrap();
        let config = StepConfig {
            dt: duration / steps as f64,
            iterations: 1,
            ..free_config()
        };
        for _ in 0..steps {
            step_cpu(&mut graph, config).unwrap();
        }
        errors.push(
            (graph.fiber(1).unwrap().motion.pose.position.x
                - base.fibers()[1].rest.position.x
                - expected)
                .abs(),
        );
    }
    assert!(
        errors[1] < errors[0] * 0.6 && errors[2] < errors[1] * 0.6,
        "errors={errors:?}"
    );
    assert!(errors[2] < 0.001);
}

#[test]
fn an_off_node_cut_piece_falls_onto_its_actual_flat_faces() {
    let mut graph = chain(2, true, false);
    let transaction = graph
        .prepare_cut(
            CutPlane::new(Vec3::new(1.371, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.3)).unwrap(),
            &[1],
            91,
        )
        .unwrap();
    let free = transaction.report().replacements[0].1[1];
    graph.commit_cut(transaction).unwrap();
    let before = graph.fiber(free).unwrap().center();
    let root = graph.fiber(0).unwrap().motion.pose;
    let config = StepConfig {
        iterations: 48,
        ..StepConfig::default()
    };
    let mut worst: f64 = 0.0;
    for _ in 0..720 {
        step_cpu(&mut graph, config).unwrap();
        let fiber = graph.fiber(free).unwrap();
        for &vertex in fiber.shape.vertices() {
            worst = worst.min(fiber.motion.pose.transform_point(vertex).y);
        }
    }
    assert_eq!(graph.fiber(0).unwrap().motion.pose, root);
    assert!(graph.fiber(free).unwrap().center().y < before.y - 1.0);
    assert!(worst > -0.002, "worst clearance={worst}");
}
