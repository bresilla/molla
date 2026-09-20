use super::*;

fn assert_state(actual: &State, expected: &State) {
    assert_eq!(
        actual.joint_q.host().unwrap(),
        expected.joint_q.host().unwrap()
    );
    assert_eq!(
        actual.joint_qd.host().unwrap(),
        expected.joint_qd.host().unwrap()
    );
    assert_eq!(
        actual.body_q.host().unwrap(),
        expected.body_q.host().unwrap()
    );
    assert_eq!(
        actual.body_qd.host().unwrap(),
        expected.body_qd.host().unwrap()
    );
    assert_eq!(
        actual.body_f.host().unwrap(),
        expected.body_f.host().unwrap()
    );
}

fn assert_island(actual: &State, local: usize, expected: &State) {
    assert_eq!(
        &actual.joint_q.host().unwrap()[8 * local..8 * local + 8],
        expected.joint_q.host().unwrap()
    );
    assert_eq!(
        &actual.joint_qd.host().unwrap()[7 * local..7 * local + 7],
        expected.joint_qd.host().unwrap()
    );
    assert_eq!(
        &actual.body_q.host().unwrap()[2 * local..2 * local + 2],
        expected.body_q.host().unwrap()
    );
    assert_eq!(
        &actual.body_qd.host().unwrap()[2 * local..2 * local + 2],
        expected.body_qd.host().unwrap()
    );
    assert_eq!(
        &actual.body_f.host().unwrap()[2 * local..2 * local + 2],
        expected.body_f.host().unwrap()
    );
}

fn contact(a: u32, b: u32) -> Contacts {
    let mut contacts = Contacts::with_capacity(1);
    contacts.count = 1;
    contacts.a_index.host_mut().unwrap()[0] = a;
    contacts.b_index.host_mut().unwrap()[0] = b;
    contacts.normal.host_mut().unwrap()[0] = Vec3::Y;
    contacts.depth.host_mut().unwrap()[0] = 0.01;
    contacts
}

#[test]
fn all_active_matches_ordinary_trajectory() {
    for rigorous in [false, true] {
        for substeps in [1, 2, 8] {
            let (model, mut expected, control) = fixture(&(0..8).collect::<Vec<_>>());
            let mut actual = expected.clone_host().unwrap();
            let mut expected_next = expected.clone_host().unwrap();
            let mut actual_next = expected.clone_host().unwrap();
            let mut solver = FeatherstoneSolver::new_cpu()
                .with_rigorous_d_solve(rigorous)
                .with_stable_motor_feedback(true)
                .with_substeps(substeps);
            let contacts = Contacts::with_capacity(0);
            for _ in 0..120 {
                solver
                    .step(
                        &model,
                        &expected,
                        &control,
                        &contacts,
                        1.0 / 120.0,
                        &mut expected_next,
                    )
                    .unwrap();
                solver
                    .step_active_articulations(
                        &model,
                        &actual,
                        &control,
                        &contacts,
                        1.0 / 120.0,
                        &mut actual_next,
                        &[true; 8],
                    )
                    .unwrap();
                assert_state(&actual_next, &expected_next);
                std::mem::swap(&mut expected, &mut expected_next);
                std::mem::swap(&mut actual, &mut actual_next);
            }
        }
    }
}

#[test]
fn selective_trajectory_matches_paused_and_resumed_simulations() {
    for rigorous in [false, true] {
        for substeps in [1, 2, 8] {
            let ids = (0..8).collect::<Vec<_>>();
            let (model, mut actual, control) = fixture(&ids);
            let mut next = actual.clone_host().unwrap();
            let mut separate: Vec<_> = ids
                .iter()
                .map(|&id| {
                    let (model, state, control) = fixture(&[id]);
                    let next = state.clone_host().unwrap();
                    (model, state, control, next)
                })
                .collect();
            let mut solver = FeatherstoneSolver::new_cpu()
                .with_rigorous_d_solve(rigorous)
                .with_stable_motor_feedback(true)
                .with_substeps(substeps);
            let contacts = Contacts::with_capacity(0);
            for step in 0..120 {
                let active: Vec<_> = ids.iter().map(|id| (step / 30 + id) % 3 == 0).collect();
                solver
                    .step_active_articulations(
                        &model,
                        &actual,
                        &control,
                        &contacts,
                        1.0 / 120.0,
                        &mut next,
                        &active,
                    )
                    .unwrap();
                std::mem::swap(&mut actual, &mut next);
                for (local, (model, state, control, next)) in separate.iter_mut().enumerate() {
                    if active[local] {
                        solver
                            .step(model, state, control, &contacts, 1.0 / 120.0, next)
                            .unwrap();
                        std::mem::swap(state, next);
                    }
                    assert_island(&actual, local, state);
                }
            }
        }
    }
}

#[test]
fn inactive_contact_has_no_reported_force_or_impulse() {
    let (model, input, control) = fixture(&[3]);
    let contacts = contact(u32::MAX, 0);
    let mut output = input.clone_host().unwrap();
    let mut solver = FeatherstoneSolver::new_cpu().with_substeps(8);
    solver
        .step_active_articulations(
            &model,
            &input,
            &control,
            &contacts,
            0.01,
            &mut output,
            &[false],
        )
        .unwrap();
    assert_state(&output, &input);
    assert_eq!(solver.contact_forces().len(), 8);
    for reading in solver.contact_forces() {
        assert!(!reading.acted);
        assert_eq!(reading.normal_force, 0.0);
        assert_eq!(reading.normal_impulse, 0.0);
        assert_eq!(reading.force_on_b, Vec3::ZERO);
        assert_eq!(reading.sample_dt, 0.01 / 8.0);
    }
    solver
        .step_active_articulations(
            &model,
            &input,
            &control,
            &contacts,
            0.01,
            &mut output,
            &[true],
        )
        .unwrap();
    assert!(
        solver
            .contact_forces()
            .iter()
            .any(|reading| reading.acted && reading.normal_impulse > 0.0)
    );
}

#[test]
fn fully_inactive_substeps_still_run_callbacks_and_can_wake_later() {
    let (model, input, control) = fixture(&[3]);
    let mut actual = input.clone_host().unwrap();
    let mut expected = input.clone_host().unwrap();
    let mut solver = FeatherstoneSolver::new_cpu().with_substeps(8);
    let mut pipeline = molla_geometry::CollisionPipeline::new();
    let mut contacts = Contacts::with_capacity(0);
    let mut calls = 0;
    solver
        .step_with_collision_active(
            &model,
            &input,
            &control,
            &mut pipeline,
            &mut contacts,
            0.01,
            &mut actual,
            |_, _| false,
            |state, _, dt, _| {
                calls += 1;
                assert_eq!(dt, 0.01 / 8.0);
                if calls <= 5 {
                    assert_state(state, &input);
                }
                Ok(vec![calls > 4])
            },
        )
        .unwrap();
    assert_eq!(calls, 8);
    solver.substeps = 4;
    solver
        .step(
            &model,
            &input,
            &control,
            &Contacts::with_capacity(0),
            0.005,
            &mut expected,
        )
        .unwrap();
    assert_state(&actual, &expected);
}

#[test]
fn inactive_contact_metadata_matches_active_geometry_and_material() {
    let (model, input, control) = fixture(&[3]);
    let mut contacts = contact(u32::MAX, 0);
    contacts.point.host_mut().unwrap()[0] = Vec3::new(1.0, 0.1, -0.2);
    contacts.normal.host_mut().unwrap()[0] = Vec3::Y * 3.0;
    contacts.friction.host_mut().unwrap()[0] = 0.37;
    let mut output = input.clone_host().unwrap();
    let mut solver = FeatherstoneSolver::new_cpu();
    solver
        .step(&model, &input, &control, &contacts, 0.001, &mut output)
        .unwrap();
    let expected = solver.contact_forces()[0].clone();
    solver
        .step_active_articulations(
            &model,
            &input,
            &control,
            &contacts,
            0.001,
            &mut output,
            &[false],
        )
        .unwrap();
    let actual = &solver.contact_forces()[0];
    assert_eq!(actual.bodies, expected.bodies);
    assert_eq!(actual.colliders, expected.colliders);
    assert_eq!(actual.point, expected.point);
    assert_eq!(actual.normal, expected.normal);
    assert_eq!(actual.penetration, expected.penetration);
    assert_eq!(actual.friction, expected.friction);
    assert_eq!(actual.sample_dt, expected.sample_dt);
    assert!(!actual.acted);
    assert_eq!(actual.force_on_b, Vec3::ZERO);
    assert_eq!(actual.normal_impulse, 0.0);
    assert_state(&output, &input);
}

#[test]
fn active_contact_state_and_readout_match_ordinary_step() {
    for substeps in [1, 2, 8] {
        let (model, input, control) = fixture(&[3, 4]);
        let contacts = contact(0, 2);
        let mut expected = input.clone_host().unwrap();
        let mut actual = input.clone_host().unwrap();
        let mut solver = FeatherstoneSolver::new_cpu().with_substeps(substeps);
        solver
            .step(&model, &input, &control, &contacts, 0.001, &mut expected)
            .unwrap();
        let readings = solver.contact_forces().to_vec();
        solver
            .step_active_articulations(
                &model,
                &input,
                &control,
                &contacts,
                0.001,
                &mut actual,
                &[true; 2],
            )
            .unwrap();
        assert_state(&actual, &expected);
        assert_eq!(solver.contact_forces().len(), substeps as usize);
        for (actual, expected) in solver.contact_forces().iter().zip(readings) {
            assert_eq!(actual.acted, expected.acted);
            assert_eq!(actual.force_on_b, expected.force_on_b);
            assert_eq!(actual.normal_force, expected.normal_force);
            assert_eq!(actual.normal_impulse, expected.normal_impulse);
            assert_eq!(actual.sample_dt, expected.sample_dt);
        }
    }
}

#[test]
fn substep_underflow_rejects_without_state_changes() {
    let (model, input, control) = fixture(&[3]);
    let mut output = input.clone_host().unwrap();
    let mut solver = FeatherstoneSolver::new_cpu().with_substeps(8);
    assert!(
        solver
            .step_active_articulations(
                &model,
                &input,
                &control,
                &Contacts::with_capacity(0),
                f64::from_bits(1),
                &mut output,
                &[true]
            )
            .is_err()
    );
    assert_state(&output, &input);
}

#[test]
fn invalid_selection_and_mixed_contacts_reject_before_output_changes() {
    let (model, input, control) = fixture(&[3, 4]);
    let contacts = contact(0, 2);
    let mut output = input.clone_host().unwrap();
    output.joint_q.host_mut().unwrap()[0] += 0.3;
    let before = output.clone_host().unwrap();
    let mut solver = FeatherstoneSolver::new_cpu();
    for (dt, mask) in [
        (0.01, vec![]),
        (0.01, vec![true]),
        (0.01, vec![true, false]),
        (0.0, vec![true; 2]),
        (f64::NAN, vec![true; 2]),
        (f64::INFINITY, vec![true; 2]),
    ] {
        assert!(
            solver
                .step_active_articulations(
                    &model,
                    &input,
                    &control,
                    &contacts,
                    dt,
                    &mut output,
                    &mask
                )
                .is_err()
        );
        assert_state(&output, &before);
        assert!(solver.contact_forces().is_empty());
    }
}

#[test]
fn immovable_contact_does_not_require_matching_activation() {
    for flags in [BodyFlags::FIXED, BodyFlags::KINEMATIC] {
        let (mut model, input, control) = fixture(&[3, 4]);
        model.body_flags.host_mut().unwrap()[2] = flags;
        let contacts = contact(2, 0);
        let mut output = input.clone_host().unwrap();
        let mut solver = FeatherstoneSolver::new_cpu();
        solver
            .step_active_articulations(
                &model,
                &input,
                &control,
                &contacts,
                0.001,
                &mut output,
                &[true, false],
            )
            .unwrap();
        assert!(solver.contact_forces()[0].acted);
        assert!(solver.contact_forces()[0].normal_force > 0.0);
        assert_eq!(
            output.body_q.host().unwrap()[2..],
            input.body_q.host().unwrap()[2..]
        );
        assert_eq!(
            output.body_qd.host().unwrap()[2..],
            input.body_qd.host().unwrap()[2..]
        );
    }
}

#[test]
fn mimic_dependencies_are_closed_and_inactive_coordinates_retained() {
    let (mut model, input, control) = fixture_with_mimic(&[3, 4], true);
    let contacts = Contacts::with_capacity(0);
    let mut output = input.clone_host().unwrap();
    let mut solver = FeatherstoneSolver::new_cpu().with_substeps(2);
    assert!(
        solver
            .step_active_articulations(
                &model,
                &input,
                &control,
                &contacts,
                0.01,
                &mut output,
                &[true, false]
            )
            .is_err()
    );
    assert_state(&output, &input);
    solver
        .step_active_articulations(
            &model,
            &input,
            &control,
            &contacts,
            0.01,
            &mut output,
            &[false; 2],
        )
        .unwrap();
    assert_state(&output, &input);
    let mut expected = input.clone_host().unwrap();
    solver
        .step(&model, &input, &control, &contacts, 0.01, &mut expected)
        .unwrap();
    solver
        .step_active_articulations(
            &model,
            &input,
            &control,
            &contacts,
            0.01,
            &mut output,
            &[true; 2],
        )
        .unwrap();
    assert_state(&output, &expected);
    model.mimic_enabled.host_mut().unwrap()[0] = 0;
    solver
        .step_active_articulations(
            &model,
            &input,
            &control,
            &contacts,
            0.01,
            &mut output,
            &[true, false],
        )
        .unwrap();
}

#[test]
fn nonphysical_contacts_do_not_join_activation_groups() {
    use molla_sim::ShapeFlags;
    let mut builder = ModelBuilder::new();
    for x in [0.0, 0.9] {
        builder.begin_articulation();
        let pose = Transform::from_translation(Vec3::X * x);
        let body = builder.add_body(BodyParams::new().with_pose(pose));
        builder
            .add_joint_free(BodyId::NONE, body, pose, Transform::IDENTITY)
            .unwrap();
        builder
            .add_shape_sphere(body, 0.5, Transform::IDENTITY)
            .unwrap();
    }
    let mut model = builder.build().unwrap();
    let input = model.state().unwrap();
    let control = model.control();
    let mut contacts = contact(0, 1);
    contacts.a_shape.host_mut().unwrap()[0] = 0;
    contacts.b_shape.host_mut().unwrap()[0] = 1;
    let mut output = input.clone_host().unwrap();
    let mut solver = FeatherstoneSolver::new_cpu();
    for flags in [ShapeFlags::COLLIDES | ShapeFlags::SENSOR, 0] {
        model.shape_flags.host_mut().unwrap()[0] = flags;
        solver
            .step_active_articulations(
                &model,
                &input,
                &control,
                &contacts,
                0.001,
                &mut output,
                &[true, false],
            )
            .unwrap();
        assert!(!solver.contact_forces()[0].acted);
        assert_eq!(solver.contact_forces()[0].normal_impulse, 0.0);
    }
    model.shape_flags.host_mut().unwrap()[0] = ShapeFlags::COLLIDES;
    for depth in [0.0, -0.01] {
        contacts.depth.host_mut().unwrap()[0] = depth;
        solver
            .step_active_articulations(
                &model,
                &input,
                &control,
                &contacts,
                0.001,
                &mut output,
                &[true, false],
            )
            .unwrap();
        assert!(!solver.contact_forces()[0].acted);
    }
    contacts.depth.host_mut().unwrap()[0] = 0.01;
    let before = output.clone_host().unwrap();
    assert!(
        solver
            .step_active_articulations(
                &model,
                &input,
                &control,
                &contacts,
                0.001,
                &mut output,
                &[true, false]
            )
            .is_err()
    );
    assert_state(&output, &before);
}

#[test]
fn malformed_activation_dependencies_return_errors() {
    let (mut model, input, control) = fixture(&[3, 4]);
    let mut output = input.clone_host().unwrap();
    let mut solver = FeatherstoneSolver::new_cpu();
    let mut contacts = contact(0, 2);
    contacts.b_index.host_mut().unwrap()[0] = 4;
    assert!(
        solver
            .step_active_articulations(
                &model,
                &input,
                &control,
                &contacts,
                0.001,
                &mut output,
                &[true; 2]
            )
            .is_err()
    );
    contacts.b_index.host_mut().unwrap()[0] = 2;
    contacts.a_shape.host_mut().unwrap()[0] = 100;
    assert!(
        solver
            .step_active_articulations(
                &model,
                &input,
                &control,
                &contacts,
                0.001,
                &mut output,
                &[true; 2]
            )
            .is_err()
    );
    contacts.count = 2;
    assert!(
        solver
            .step_active_articulations(
                &model,
                &input,
                &control,
                &contacts,
                0.001,
                &mut output,
                &[true; 2]
            )
            .is_err()
    );
    contacts.clear();
    model.articulation_start.host_mut().unwrap()[1] = 100;
    assert!(
        solver
            .step_active_articulations(
                &model,
                &input,
                &control,
                &contacts,
                0.001,
                &mut output,
                &[true; 2]
            )
            .is_err()
    );
    model.articulation_start.host_mut().unwrap()[1] = 2;
    model.mimic_count = 1;
    assert!(
        solver
            .step_active_articulations(
                &model,
                &input,
                &control,
                &contacts,
                0.001,
                &mut output,
                &[true; 2]
            )
            .is_err()
    );
    assert_state(&output, &input);
}

#[test]
fn empty_model_and_implicit_single_articulation_follow_normal_step() {
    let (mut model, input, control) = fixture(&[3]);
    model.articulation_count = 0;
    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_cpu();
    let mut actual = input.clone_host().unwrap();
    let mut expected = input.clone_host().unwrap();
    solver
        .step(&model, &input, &control, &contacts, 0.001, &mut expected)
        .unwrap();
    solver
        .step_active_articulations(
            &model,
            &input,
            &control,
            &contacts,
            0.001,
            &mut actual,
            &[true],
        )
        .unwrap();
    assert_state(&actual, &expected);
    solver
        .step_active_articulations(
            &model,
            &input,
            &control,
            &contacts,
            0.001,
            &mut actual,
            &[false],
        )
        .unwrap();
    assert_state(&actual, &input);
    let model = ModelBuilder::new().build().unwrap();
    let input = model.state().unwrap();
    let mut output = input.clone_host().unwrap();
    solver
        .step_active_articulations(
            &model,
            &input,
            &model.control(),
            &contacts,
            0.001,
            &mut output,
            &[],
        )
        .unwrap();
    assert_state(&output, &input);
}

#[test]
fn dependency_snapshot_selects_complete_coupled_articulations() {
    use molla_sim::runtime::{BodyDesc, RigidScene};
    use molla_solvers::rigid_islands::RigidIslands;
    let mut scene = RigidScene::new().unwrap();
    let handles: Vec<_> = (0..3)
        .map(|i| {
            scene
                .insert_body(BodyDesc {
                    params: BodyParams::new().with_position(Vec3::X * i as f64),
                    ..Default::default()
                })
                .unwrap()
        })
        .collect();
    let graph = RigidIslands::build(&scene, [[Some(handles[0]), Some(handles[1])]]).unwrap();
    let wake = graph.affected_bodies(&[handles[0]]).unwrap();
    assert_eq!(wake, handles[..2]);
    let mut mask = vec![false; scene.model().articulation_count];
    for group in graph.groups() {
        if group.bodies.iter().any(|body| wake.contains(body)) {
            for &art in &group.articulations {
                mask[art] = true;
            }
        }
    }
    assert_eq!(mask.iter().filter(|&&active| active).count(), 2);
    let input = scene.state();
    let mut output = input.clone_host().unwrap();
    let mut solver = FeatherstoneSolver::new_cpu();
    solver
        .step_active_articulations(
            scene.model(),
            input,
            scene.control(),
            &Contacts::with_capacity(0),
            0.001,
            &mut output,
            &mask,
        )
        .unwrap();
    let moved = output
        .body_q
        .host()
        .unwrap()
        .iter()
        .zip(input.body_q.host().unwrap())
        .filter(|(after, before)| after.position.y < before.position.y)
        .count();
    assert_eq!(moved, 2);
}

#[test]
#[ignore = "CPU activation benchmark; run in release without concurrent builds"]
fn selective_articulation_scaling() {
    for count in [1, 32, 128, 512] {
        let (model, input, control) = fixture(&(0..count).collect::<Vec<_>>());
        let mut output = input.clone_host().unwrap();
        let contacts = Contacts::with_capacity(0);
        let mut solver = FeatherstoneSolver::new_cpu()
            .with_rigorous_d_solve(true)
            .with_stable_motor_feedback(true);
        let masks: Vec<Vec<_>> = [0, 1, count]
            .into_iter()
            .map(|enabled| (0..count).map(|i| i < enabled).collect())
            .collect();
        let mut samples = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for i in 0..100 {
            for lane in 0..4 {
                let lane = if i % 2 == 0 { lane } else { 3 - lane };
                let start = std::time::Instant::now();
                if lane == 0 {
                    solver
                        .step(
                            &model,
                            &input,
                            &control,
                            &contacts,
                            1.0 / 120.0,
                            &mut output,
                        )
                        .unwrap();
                } else {
                    solver
                        .step_active_articulations(
                            &model,
                            &input,
                            &control,
                            &contacts,
                            1.0 / 120.0,
                            &mut output,
                            &masks[lane - 1],
                        )
                        .unwrap();
                }
                if i >= 20 {
                    samples[lane].push(start.elapsed().as_secs_f64() * 1000.0);
                }
            }
        }
        for (lane, mut samples) in samples.into_iter().enumerate() {
            samples.sort_by(f64::total_cmp);
            eprintln!(
                "activation scaling: articulations={count} mode={} active={} median_ms={:.6} p95_ms={:.6}",
                if lane == 0 { "ordinary" } else { "selected" },
                [count, 0, 1, count][lane],
                samples[40],
                samples[76]
            );
        }
    }
}
