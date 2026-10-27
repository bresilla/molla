use super::*;

fn chains(sizes: &[(usize, usize)]) -> (Model, State, Control) {
    let mut builder = ModelBuilder::new();
    for &(seed, count) in sizes {
        builder.begin_articulation();
        let origin = Transform::from_translation(Vec3::new(seed as f64 * 4.0, 2.0, 0.0));
        let mut parent = BodyId::NONE;
        for local in 0..count {
            let body = builder.add_body(
                BodyParams::new()
                    .with_mass(1.0 + local as f64)
                    .with_inertia(Mat3::IDENTITY * (0.1 + local as f64 * 0.2)),
            );
            if local == 0 {
                builder
                    .add_joint_free(parent, body, origin, Transform::IDENTITY)
                    .unwrap();
            } else {
                let frame = Transform::from_translation(Vec3::new(0.1, -0.2, 0.0));
                let joint = if local % 2 == 0 {
                    builder
                        .add_joint_prismatic(parent, body, frame, Transform::IDENTITY, Vec3::Y)
                        .unwrap()
                } else {
                    builder
                        .add_joint_revolute(parent, body, frame, Transform::IDENTITY, Vec3::Z)
                        .unwrap()
                };
                builder.set_joint_limits(joint, &[-0.2], &[0.2]).unwrap();
            }
            parent = body;
        }
    }
    let mut model = builder.build().unwrap();
    for (art, &(seed, _)) in sizes.iter().enumerate() {
        if seed % 2 == 0 {
            let joint = model.articulation_start.host().unwrap()[art] as usize;
            let body = model.joint_child.host().unwrap()[joint] as usize;
            model.body_flags.host_mut().unwrap()[body] = BodyFlags::FIXED;
        }
    }
    let mut state = model.state().unwrap();
    let mut control = model.control();
    for (art, &(seed, count)) in sizes.iter().enumerate() {
        let begin = model.articulation_start.host().unwrap()[art] as usize;
        for local in 0..count {
            let dof = model.joint_dof_offset.host().unwrap()[begin + local] as usize;
            control.joint_target_mode.host_mut().unwrap()[dof] = JointTargetMode::Velocity as u32;
            control.joint_target_vel.host_mut().unwrap()[dof] = (seed + 1) as f64 * 0.1;
            control.joint_target_kd.host_mut().unwrap()[dof] = 2.0;
            control.joint_motor_max_force.host_mut().unwrap()[dof] = 10.0;
        }
    }
    molla_sim::eval_fk(&model, &mut state).unwrap();
    molla_sim::eval_body_velocities(&model, &mut state).unwrap();
    (model, state, control)
}

#[test]
fn heterogeneous_articulations_match_separate_worlds_through_wake_changes() {
    let sizes = [(0, 1), (1, 7), (2, 3), (3, 11), (4, 2)];
    for rigorous in [false, true] {
        for substeps in [1, 8] {
            let (model, mut state, control) = chains(&sizes);
            let mut next = state.clone_host().unwrap();
            let mut separate: Vec<_> = sizes
                .iter()
                .map(|&pair| {
                    let (model, state, control) = chains(&[pair]);
                    let next = state.clone_host().unwrap();
                    (model, state, control, next)
                })
                .collect();
            let mut solver = FeatherstoneSolver::new_cpu()
                .with_substeps(substeps)
                .with_rigorous_d_solve(rigorous)
                .with_stable_motor_feedback(true);
            let contacts = Contacts::with_capacity(0);
            for step in 0..80 {
                let active: Vec<_> = (0..sizes.len())
                    .map(|art| (step / 10 + art) % 3 != 0)
                    .collect();
                solver
                    .step_active_articulations(
                        &model, &state, &control, &contacts, 0.005, &mut next, &active,
                    )
                    .unwrap();
                std::mem::swap(&mut state, &mut next);
                let mut body = 0;
                for (art, (local_model, local, control, next)) in separate.iter_mut().enumerate() {
                    if active[art] {
                        solver
                            .step(local_model, local, control, &contacts, 0.005, next)
                            .unwrap();
                        std::mem::swap(local, next);
                    }
                    let start = model.articulation_start.host().unwrap()[art] as usize;
                    let end = model.articulation_start.host().unwrap()[art + 1] as usize;
                    let coords = model.joint_coord_offset.host().unwrap();
                    let dofs = model.joint_dof_offset.host().unwrap();
                    assert_eq!(
                        &state.joint_q.host().unwrap()
                            [coords[start] as usize..coords[end] as usize],
                        local.joint_q.host().unwrap()
                    );
                    assert_eq!(
                        &state.joint_qd.host().unwrap()[dofs[start] as usize..dofs[end] as usize],
                        local.joint_qd.host().unwrap()
                    );
                    let count = sizes[art].1;
                    assert_eq!(
                        &state.body_q.host().unwrap()[body..body + count],
                        local.body_q.host().unwrap()
                    );
                    assert_eq!(
                        &state.body_qd.host().unwrap()[body..body + count],
                        local.body_qd.host().unwrap()
                    );
                    body += count;
                }
            }
        }
    }
}

fn assert_state_bits(a: &State, b: &State) {
    let equal = |a: f64, b: f64| assert_eq!(a.to_bits(), b.to_bits());
    for (a, b) in [
        (a.joint_q.host().unwrap(), b.joint_q.host().unwrap()),
        (a.joint_qd.host().unwrap(), b.joint_qd.host().unwrap()),
    ] {
        assert_eq!(a.len(), b.len());
        for (&a, &b) in a.iter().zip(b) {
            equal(a, b);
        }
    }
    for (a, b) in a
        .body_q
        .host()
        .unwrap()
        .iter()
        .zip(b.body_q.host().unwrap())
    {
        for (a, b) in a.position.to_array().into_iter().zip(b.position.to_array()) {
            equal(a, b);
        }
        for (a, b) in a.rotation.to_array().into_iter().zip(b.rotation.to_array()) {
            equal(a, b);
        }
    }
    for (a, b) in [
        (a.body_qd.host().unwrap(), b.body_qd.host().unwrap()),
        (a.body_f.host().unwrap(), b.body_f.host().unwrap()),
    ] {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            for (a, b) in a.linear.to_array().into_iter().zip(b.linear.to_array()) {
                equal(a, b);
            }
            for (a, b) in a.angular.to_array().into_iter().zip(b.angular.to_array()) {
                equal(a, b);
            }
        }
    }
}

#[test]
fn reused_solver_matches_fresh_solver_across_models_and_selection() {
    let layouts: &[&[(usize, usize)]] = &[
        &[(1, 11)],
        &[(0, 1), (1, 7), (2, 3), (3, 11), (4, 2)],
        &[],
        &[(3, 1)],
        &[(1, 2), (2, 1)],
        &[(1, 17)],
    ];
    let contacts = Contacts::with_capacity(0);
    for rigorous in [false, true] {
        for substeps in [1, 4] {
            let new_solver = || {
                FeatherstoneSolver::new_cpu()
                    .with_substeps(substeps)
                    .with_rigorous_d_solve(rigorous)
                    .with_stable_motor_feedback(true)
            };
            let mut reused = new_solver();
            for &layout in layouts.iter().cycle().take(layouts.len() * 2) {
                let (model, mut state, control) = chains(layout);
                let mut actual = state.clone_host().unwrap();
                let mut expected = state.clone_host().unwrap();
                for step in 0..12 {
                    let selected: Vec<_> = (0..model.articulation_count)
                        .map(|art| step % 5 != 0 && (step + art) % 3 != 0)
                        .collect();
                    assert!(
                        reused
                            .step_active_articulations(
                                &model,
                                &state,
                                &control,
                                &contacts,
                                f64::NAN,
                                &mut actual,
                                &selected,
                            )
                            .is_err()
                    );
                    let mut fresh = new_solver();
                    if step % 4 == 0 {
                        reused
                            .step(&model, &state, &control, &contacts, 0.002, &mut actual)
                            .unwrap();
                        fresh
                            .step(&model, &state, &control, &contacts, 0.002, &mut expected)
                            .unwrap();
                    } else {
                        reused
                            .step_active_articulations(
                                &model,
                                &state,
                                &control,
                                &contacts,
                                0.002,
                                &mut actual,
                                &selected,
                            )
                            .unwrap();
                        fresh
                            .step_active_articulations(
                                &model,
                                &state,
                                &control,
                                &contacts,
                                0.002,
                                &mut expected,
                                &selected,
                            )
                            .unwrap();
                    }
                    assert_state_bits(&actual, &expected);
                    std::mem::swap(&mut state, &mut actual);
                }
            }
        }
    }
}
