use molla_core::BodyId;
use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, BodyHandle, BodyKind, JointDesc, RigidScene};
use molla_sim::{BodyFlags, BodyParams, JointType, MassProperties, Model, State};
use molla_solvers::rigid_impulse::{ImpactConstraint, ImpactContact, ImpactRow, apply_impact};

fn body(
    scene: &mut RigidScene,
    mass: f64,
    position: Vec3,
    velocity: Vec3,
    kind: BodyKind,
) -> BodyHandle {
    scene
        .insert_body(BodyDesc {
            kind,
            params: BodyParams::new().with_position(position),
            velocity: SpatialVector::new(velocity, Vec3::ZERO),
            additional_mass: Some(MassProperties {
                mass,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap()
}

fn contact(
    bodies: [Option<BodyId>; 2],
    point: Vec3,
    normal: Vec3,
    restitution: f64,
) -> ImpactConstraint {
    ImpactConstraint::Contact(ImpactContact {
        bodies,
        point,
        normal,
        restitution,
        static_friction: 0.,
        dynamic_friction: 0.,
        surface_velocity: [Vec3::ZERO; 2],
    })
}

fn near(actual: Vec3, expected: Vec3) {
    assert!(
        (actual - expected).length() <= 1e-8 * (1. + expected.length()),
        "{actual:?} != {expected:?}"
    );
}

fn energy(model: &Model, state: &State) -> f64 {
    (0..model.body_count)
        .filter(|&b| model.body_flags.host().unwrap()[b] & BodyFlags::IMMOVABLE == 0)
        .map(|b| {
            let velocity = state.body_qd.host().unwrap()[b];
            let omega = state.body_q.host().unwrap()[b].rotation.inverse() * velocity.angular;
            0.5 * (model.body_mass.host().unwrap()[b] * velocity.linear.length_squared()
                + omega.dot(model.body_inertia.host().unwrap()[b] * omega))
        })
        .sum()
}

#[test]
fn bounded_feedback_keeps_physical_rates_and_work_and_rejects_invalid_edits() {
    for mass in [0.01, 1.0, 100.0] {
        for diagonal in [0.0, 0.5 / mass, 2.0 / mass] {
            for bias in [-4.0, 0.0, 4.0] {
                for bounds in [
                    [-0.7 * mass, 0.7 * mass],
                    [0.2 * mass; 2],
                    [-f64::INFINITY, f64::INFINITY],
                ] {
                    let mut scene = RigidScene::new().unwrap();
                    let body = body(
                        &mut scene,
                        mass,
                        Vec3::ZERO,
                        Vec3::X * 3.0,
                        BodyKind::Dynamic,
                    );
                    let id = scene.body_index(body).unwrap();
                    let mut state = scene.state().clone_host().unwrap();
                    let before = state.clone_host().unwrap();
                    let row = ImpactRow {
                        ends: vec![(id, SpatialVector::new(Vec3::X, Vec3::ZERO))],
                        velocity_bias: bias,
                        ..Default::default()
                    };
                    let result = apply_impact(
                        scene.model(),
                        &mut state,
                        &[ImpactConstraint::Bounded {
                            row: row.clone(),
                            impulses: bounds,
                            response_diagonal: diagonal,
                        }],
                    )
                    .unwrap();
                    let impulse =
                        (-(3.0 + bias) / (1.0 / mass + diagonal)).clamp(bounds[0], bounds[1]);
                    near(
                        state.body_qd.host().unwrap()[id.index()].linear,
                        Vec3::X * (3.0 + impulse / mass),
                    );
                    assert!(
                        (result.solution.impulses[0] - impulse).abs()
                            < 1e-9 * (1.0 + impulse.abs())
                    );
                    assert!(
                        (energy(scene.model(), &state)
                            - energy(scene.model(), &before)
                            - result.relative_work[0]
                            - result.boundary_work[0])
                            .abs()
                            < 1e-8 * (1.0 + mass)
                    );
                    assert_eq!(state.body_q.host().unwrap(), before.body_q.host().unwrap());
                    for invalid in [-1.0, f64::NAN, f64::INFINITY] {
                        let before = state.clone_host().unwrap();
                        assert!(
                            apply_impact(
                                scene.model(),
                                &mut state,
                                &[ImpactConstraint::Bounded {
                                    row: row.clone(),
                                    impulses: bounds,
                                    response_diagonal: invalid,
                                }]
                            )
                            .is_err()
                        );
                        assert_eq!(
                            state.joint_qd.host().unwrap(),
                            before.joint_qd.host().unwrap()
                        );
                        assert_eq!(
                            state.body_qd.host().unwrap(),
                            before.body_qd.host().unwrap()
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn two_body_rebound_momentum_endpoint_exchange_and_force_preservation() {
    for (ma, mb) in [(1., 1.), (1., 5.), (1e-3, 1e3)] {
        for restitution in [0., 0.5, 1.] {
            for reverse in [false, true] {
                let mut scene = RigidScene::new().unwrap();
                let a = body(&mut scene, ma, Vec3::ZERO, Vec3::X, BodyKind::Dynamic);
                let b = body(&mut scene, mb, Vec3::X * 2., Vec3::ZERO, BodyKind::Dynamic);
                let ia = scene.body_index(a).unwrap();
                let ib = scene.body_index(b).unwrap();
                let mut state = scene.state().clone_host().unwrap();
                state.body_f.host_mut().unwrap()[ia.index()] =
                    SpatialVector::new(Vec3::Y * 7., Vec3::Z);
                let before = state.clone_host().unwrap();
                let row = if reverse {
                    contact([Some(ib), Some(ia)], Vec3::X, -Vec3::X, restitution)
                } else {
                    contact([Some(ia), Some(ib)], Vec3::X, Vec3::X, restitution)
                };
                let result = apply_impact(scene.model(), &mut state, &[row]).unwrap();
                let va = state.body_qd.host().unwrap()[ia.index()].linear;
                let vb = state.body_qd.host().unwrap()[ib.index()].linear;
                near(va, Vec3::X * ((ma - restitution * mb) / (ma + mb)));
                near(vb, Vec3::X * ((1. + restitution) * ma / (ma + mb)));
                near(va * ma + vb * mb, Vec3::X * ma);
                let change = energy(scene.model(), &state) - energy(scene.model(), &before);
                assert!((change - result.relative_work[0] - result.boundary_work[0]).abs() < 1e-8);
                assert!(change <= 1e-8);
                assert_eq!(state.body_q.host().unwrap(), before.body_q.host().unwrap());
                assert_eq!(
                    state.joint_q.host().unwrap(),
                    before.joint_q.host().unwrap()
                );
                assert_eq!(state.body_f.host().unwrap(), before.body_f.host().unwrap());
            }
        }
    }
}

#[test]
fn off_center_rotated_com_and_tensor_match_analytic_impulse() {
    let inertia = Mat3::from_cols(
        Vec3::new(2., 0.3, 0.),
        Vec3::new(0.3, 1., 0.),
        Vec3::new(0., 0., 1.5),
    );
    for shift in [Vec3::ZERO, Vec3::new(1024., -512., 256.)] {
        for rotation in [
            Quat::IDENTITY,
            Quat::from_rotation_z(0.7) * Quat::from_rotation_y(0.4),
        ] {
            let mut scene = RigidScene::new().unwrap();
            let com = Vec3::new(0.2, -0.3, 0.1);
            let b = scene
                .insert_body(BodyDesc {
                    params: BodyParams {
                        pose: Transform {
                            position: shift,
                            rotation,
                        },
                        ..Default::default()
                    },
                    velocity: SpatialVector::new(rotation * -Vec3::X, Vec3::ZERO),
                    additional_mass: Some(MassProperties {
                        mass: 2.,
                        local_com: com,
                        inertia,
                    }),
                    ..Default::default()
                })
                .unwrap();
            let id = scene.body_index(b).unwrap();
            let arm = rotation * (Vec3::Y * 0.4);
            let normal = rotation * Vec3::X;
            let point = shift + rotation * com + arm;
            let inverse = Mat3::from_quat(rotation)
                * inertia.inverse()
                * Mat3::from_quat(rotation).transpose();
            let moment = arm.cross(normal);
            let impulse = 1.5 / (0.5 + moment.dot(inverse * moment));
            let mut state = scene.state().clone_host().unwrap();
            let before = energy(scene.model(), &state);
            let result = apply_impact(
                scene.model(),
                &mut state,
                &[contact([None, Some(id)], point, normal, 0.5)],
            )
            .unwrap();
            near(result.contact_impulses[0].unwrap(), normal * impulse);
            let velocity = state.body_qd.host().unwrap()[id.index()];
            near(velocity.linear, -normal + normal * (impulse / 2.));
            near(velocity.angular, inverse * moment * impulse);
            let change = energy(scene.model(), &state) - before;
            assert!((change - result.relative_work[0] - result.boundary_work[0]).abs() < 1e-8);
        }
    }
}

#[test]
fn fixed_tree_impact_transfers_through_the_articulation() {
    let mut scene = RigidScene::new().unwrap();
    let parent = body(&mut scene, 1., Vec3::ZERO, Vec3::ZERO, BodyKind::Dynamic);
    let child = body(&mut scene, 3., Vec3::X * 2., Vec3::ZERO, BodyKind::Dynamic);
    scene
        .insert_joint(JointDesc {
            parent: Some(parent),
            child,
            kind: JointType::Fixed,
            axis: Vec3::X,
            frame_parent: Transform::from_translation(Vec3::X * 2.),
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    scene
        .set_body_velocity(parent, SpatialVector::new(-Vec3::X, Vec3::ZERO))
        .unwrap();
    let mut state = scene.state().clone_host().unwrap();
    let result = apply_impact(
        scene.model(),
        &mut state,
        &[contact(
            [None, scene.body_index(child)],
            Vec3::X * 3.,
            Vec3::X,
            0.5,
        )],
    )
    .unwrap();
    near(result.contact_impulses[0].unwrap(), Vec3::X * 6.);
    for body in [parent, child] {
        let v = state.body_qd.host().unwrap()[scene.body_index(body).unwrap().index()];
        near(v.linear, Vec3::X * 0.5);
        near(v.angular, Vec3::ZERO);
    }
}

#[test]
fn world_hinge_uses_pivot_inertia_and_preserves_the_locked_motion() {
    let mut scene = RigidScene::new().unwrap();
    let child = body(&mut scene, 2., Vec3::Y, Vec3::ZERO, BodyKind::Dynamic);
    scene
        .insert_joint(JointDesc {
            parent: None,
            child,
            kind: JointType::Revolute,
            axis: Vec3::Z,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::from_translation(-Vec3::Y),
        })
        .unwrap();
    scene
        .set_body_velocity(child, SpatialVector::new(-Vec3::X, Vec3::Z))
        .unwrap();
    let mut state = scene.state().clone_host().unwrap();
    let result = apply_impact(
        scene.model(),
        &mut state,
        &[contact(
            [None, scene.body_index(child)],
            Vec3::Y,
            Vec3::X,
            0.5,
        )],
    )
    .unwrap();
    near(result.contact_impulses[0].unwrap(), Vec3::X * 4.5);
    let v = state.body_qd.host().unwrap()[scene.body_index(child).unwrap().index()];
    near(v.linear, Vec3::X * 0.5);
    near(v.angular, -Vec3::Z * 0.5);
}

#[test]
fn moving_support_work_explains_dynamic_energy_gain() {
    let mut scene = RigidScene::new().unwrap();
    let support = body(&mut scene, 1., Vec3::ZERO, Vec3::X, BodyKind::Kinematic);
    let ball = body(&mut scene, 2., Vec3::X, -Vec3::X, BodyKind::Dynamic);
    let mut state = scene.state().clone_host().unwrap();
    let before = energy(scene.model(), &state);
    let result = apply_impact(
        scene.model(),
        &mut state,
        &[contact(
            [scene.body_index(support), scene.body_index(ball)],
            Vec3::X * 0.5,
            Vec3::X,
            0.5,
        )],
    )
    .unwrap();
    near(
        state.body_qd.host().unwrap()[scene.body_index(support).unwrap().index()].linear,
        Vec3::X,
    );
    near(
        state.body_qd.host().unwrap()[scene.body_index(ball).unwrap().index()].linear,
        Vec3::X * 2.,
    );
    assert!((result.relative_work[0] + 3.).abs() < 1e-8);
    assert!((result.boundary_work[0] - 6.).abs() < 1e-8);
    assert!((energy(scene.model(), &state) - before - 3.).abs() < 1e-8);
}

#[test]
fn inherited_kinematic_velocity_uses_physical_not_relative_joint_energy() {
    let mut scene = RigidScene::new().unwrap();
    let support = body(&mut scene, 1., Vec3::ZERO, Vec3::X, BodyKind::Kinematic);
    let child = body(&mut scene, 2., Vec3::ZERO, Vec3::ZERO, BodyKind::Dynamic);
    scene
        .insert_joint(JointDesc {
            parent: Some(support),
            child,
            kind: JointType::Prismatic,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    scene
        .set_body_velocity(child, SpatialVector::new(-Vec3::X, Vec3::ZERO))
        .unwrap();
    let mut state = scene.state().clone_host().unwrap();
    let before = energy(scene.model(), &state);
    let result = apply_impact(
        scene.model(),
        &mut state,
        &[contact(
            [None, scene.body_index(child)],
            Vec3::ZERO,
            Vec3::X,
            0.5,
        )],
    )
    .unwrap();
    near(
        state.body_qd.host().unwrap()[scene.body_index(child).unwrap().index()].linear,
        Vec3::X * 0.5,
    );
    let change = energy(scene.model(), &state) - before;
    assert!((change + 0.75).abs() < 1e-8);
    assert!((change - result.relative_work[0] - result.boundary_work[0]).abs() < 1e-8);
}

#[test]
fn scalar_joint_row_couples_to_contact_and_failures_leave_state_unchanged() {
    let mut scene = RigidScene::new().unwrap();
    let b = body(&mut scene, 1., Vec3::ZERO, -Vec3::X, BodyKind::Dynamic);
    let id = scene.body_index(b).unwrap();
    let row = ImpactRow {
        direct: None,
        ends: vec![(id, SpatialVector::new(Vec3::new(1., 1., 0.), Vec3::ZERO))],
        velocity_bias: 0.,
    };
    let mut state = scene.state().clone_host().unwrap();
    let result = apply_impact(
        scene.model(),
        &mut state,
        &[
            contact([None, Some(id)], Vec3::ZERO, Vec3::X, 0.5),
            ImpactConstraint::Bilateral(row),
        ],
    )
    .unwrap();
    near(
        state.body_qd.host().unwrap()[id.index()].linear,
        Vec3::new(0.5, -0.5, 0.),
    );
    near(result.contact_impulses[0].unwrap(), Vec3::X * 2.);
    for invalid in [
        contact([None, Some(id)], Vec3::ZERO, Vec3::X * 2., 0.5),
        contact([None, Some(id)], Vec3::ZERO, Vec3::X, 1.1),
        contact(
            [None, Some(BodyId::from_index(999))],
            Vec3::ZERO,
            Vec3::X,
            0.5,
        ),
        ImpactConstraint::Bilateral(ImpactRow {
            direct: None,
            ends: Vec::new(),
            velocity_bias: 1.,
        }),
    ] {
        let before = state.clone_host().unwrap();
        assert!(apply_impact(scene.model(), &mut state, &[invalid]).is_err());
        assert_eq!(
            state.joint_qd.host().unwrap(),
            before.joint_qd.host().unwrap()
        );
        assert_eq!(
            state.body_qd.host().unwrap(),
            before.body_qd.host().unwrap()
        );
        assert_eq!(state.body_q.host().unwrap(), before.body_q.host().unwrap());
        assert_eq!(state.body_f.host().unwrap(), before.body_f.host().unwrap());
    }
}

#[test]
fn oblique_friction_impacts_dissipate_with_world_frame_covariance() {
    let mut reference = std::collections::BTreeMap::new();
    for rotation in [
        Quat::IDENTITY,
        Quat::from_rotation_y(0.7) * Quat::from_rotation_x(0.5),
    ] {
        for shift in [Vec3::ZERO, Vec3::new(2048., -1024., 512.)] {
            for mu in [0.1, 0.5, 2.] {
                for restitution in [0., 0.5, 1.] {
                    let mut scene = RigidScene::new().unwrap();
                    let local_com = Vec3::new(0.2, 0.3, -0.1);
                    let velocity = SpatialVector::new(
                        rotation * Vec3::new(3., -1., -2.),
                        rotation * Vec3::new(0.2, -0.3, 0.4),
                    );
                    let b = scene
                        .insert_body(BodyDesc {
                            params: BodyParams {
                                pose: Transform {
                                    position: shift,
                                    rotation,
                                },
                                ..Default::default()
                            },
                            velocity,
                            additional_mass: Some(MassProperties {
                                mass: 2.,
                                local_com,
                                inertia: Mat3::from_diagonal(Vec3::new(1., 1.5, 2.)),
                            }),
                            ..Default::default()
                        })
                        .unwrap();
                    let id = scene.body_index(b).unwrap();
                    let arm = rotation * Vec3::new(0.5, 1., 0.3);
                    let normal = rotation * Vec3::Y;
                    let point = shift + rotation * local_com + arm;
                    let mut state = scene.state().clone_host().unwrap();
                    let initial_energy = energy(scene.model(), &state);
                    let incoming = (velocity.linear + velocity.angular.cross(arm)).dot(normal);
                    let result = apply_impact(
                        scene.model(),
                        &mut state,
                        &[ImpactConstraint::Contact(ImpactContact {
                            bodies: [None, Some(id)],
                            point,
                            normal,
                            restitution,
                            static_friction: mu * 1.5,
                            dynamic_friction: mu,
                            surface_velocity: [Vec3::ZERO; 2],
                        })],
                    )
                    .unwrap();
                    let after = state.body_qd.host().unwrap()[id.index()];
                    let outgoing = (after.linear + after.angular.cross(arm)).dot(normal);
                    assert!((outgoing + restitution * incoming).abs() < 1e-8);
                    let impulse = result.contact_impulses[0].unwrap();
                    let normal_impulse = impulse.dot(normal);
                    let tangent_impulse = impulse - normal * normal_impulse;
                    assert!(
                        normal_impulse >= 0.
                            && tangent_impulse.length() <= mu * normal_impulse * (1. + 1e-8)
                    );
                    let change = energy(scene.model(), &state) - initial_energy;
                    assert!(change <= 1e-8, "energy increase {change}");
                    assert!(
                        (change - result.relative_work[0] - result.boundary_work[0]).abs() < 1e-8
                    );
                    near(after.linear - velocity.linear, impulse / 2.);
                    let local = (
                        rotation.inverse() * after.linear,
                        rotation.inverse() * after.angular,
                    );
                    let old = reference
                        .entry((mu.to_bits(), restitution.to_bits()))
                        .or_insert(local);
                    near(local.0, old.0);
                    near(local.1, old.1);
                }
            }
        }
    }
}

#[test]
fn moving_surface_transfers_work_and_ignores_normal_belt_velocity() {
    let mut scene = RigidScene::new().unwrap();
    let b = body(&mut scene, 1., Vec3::ZERO, -Vec3::Y, BodyKind::Dynamic);
    let id = scene.body_index(b).unwrap();
    let mut reference = None;
    for normal_belt in [0., 100.] {
        let mut state = scene.state().clone_host().unwrap();
        let initial = energy(scene.model(), &state);
        let result = apply_impact(
            scene.model(),
            &mut state,
            &[ImpactConstraint::Contact(ImpactContact {
                bodies: [None, Some(id)],
                point: Vec3::ZERO,
                normal: Vec3::Y,
                restitution: 1.,
                static_friction: 0.5,
                dynamic_friction: 0.5,
                surface_velocity: [Vec3::X * 3. + Vec3::Y * normal_belt, Vec3::ZERO],
            })],
        )
        .unwrap();
        let velocity = state.body_qd.host().unwrap()[id.index()].linear;
        near(velocity, Vec3::new(1., 1., 0.));
        assert!((result.relative_work[0] + 2.5).abs() < 1e-8);
        assert!((result.boundary_work[0] - 3.).abs() < 1e-8);
        assert!((energy(scene.model(), &state) - initial - 0.5).abs() < 1e-8);
        if let Some(old) = reference {
            near(velocity, old);
        }
        reference = Some(velocity);
    }
}

#[test]
fn simultaneous_manifold_contacts_are_order_independent_in_body_response() {
    let mut scene = RigidScene::new().unwrap();
    let b = body(&mut scene, 2., Vec3::ZERO, Vec3::ZERO, BodyKind::Dynamic);
    scene
        .set_body_velocity(b, SpatialVector::new(-Vec3::Y, Vec3::Z * 0.2))
        .unwrap();
    let id = scene.body_index(b).unwrap();
    let mut contacts = [
        contact([None, Some(id)], -Vec3::X, Vec3::Y, 0.5),
        contact([None, Some(id)], Vec3::X, Vec3::Y, 0.5),
    ];
    for _ in 0..2 {
        let mut state = scene.state().clone_host().unwrap();
        let before = energy(scene.model(), &state);
        let result = apply_impact(scene.model(), &mut state, &contacts).unwrap();
        let velocity = state.body_qd.host().unwrap()[id.index()];
        near(velocity.linear, Vec3::Y * 0.5);
        near(velocity.angular, -Vec3::Z * 0.1);
        near(
            result.contact_impulses.iter().flatten().copied().sum(),
            Vec3::Y * 3.,
        );
        let recorded: f64 = result
            .relative_work
            .iter()
            .chain(&result.boundary_work)
            .sum();
        assert!((energy(scene.model(), &state) - before - recorded).abs() < 1e-8);
        contacts.reverse();
    }
}

#[test]
fn off_center_two_body_friction_conserves_linear_and_angular_momentum() {
    let mut scene = RigidScene::new().unwrap();
    let a = body(&mut scene, 1., -Vec3::X, Vec3::X, BodyKind::Dynamic);
    let b = body(&mut scene, 3., Vec3::X, -Vec3::X, BodyKind::Dynamic);
    let mut state = scene.state().clone_host().unwrap();
    let before = energy(scene.model(), &state);
    let result = apply_impact(
        scene.model(),
        &mut state,
        &[ImpactConstraint::Contact(ImpactContact {
            bodies: [scene.body_index(a), scene.body_index(b)],
            point: Vec3::Y * 0.3,
            normal: Vec3::X,
            restitution: 0.5,
            static_friction: 0.6,
            dynamic_friction: 0.3,
            surface_velocity: [Vec3::ZERO; 2],
        })],
    )
    .unwrap();
    let mut linear = Vec3::ZERO;
    let mut angular = Vec3::ZERO;
    for (body, mass) in [(a, 1.), (b, 3.)] {
        let i = scene.body_index(body).unwrap().index();
        let v = state.body_qd.host().unwrap()[i];
        linear += v.linear * mass;
        angular += state.body_q.host().unwrap()[i]
            .position
            .cross(v.linear * mass)
            + v.angular;
    }
    near(linear, -Vec3::X * 2.);
    near(angular, Vec3::ZERO);
    let change = energy(scene.model(), &state) - before;
    assert!(change <= 1e-8);
    assert!((change - result.relative_work[0] - result.boundary_work[0]).abs() < 1e-8);
}

#[test]
fn nonnegative_scalar_response_and_late_failure_are_transactional() {
    let mut scene = RigidScene::new().unwrap();
    let b = body(
        &mut scene,
        1.,
        Vec3::ZERO,
        -Vec3::X - Vec3::Y,
        BodyKind::Dynamic,
    );
    let id = scene.body_index(b).unwrap();
    let impact = contact([None, Some(id)], Vec3::ZERO, Vec3::X, 0.5);
    let row = ImpactRow {
        direct: None,
        ends: vec![(id, SpatialVector::new(Vec3::Y, Vec3::ZERO))],
        velocity_bias: 0.,
    };
    let mut state = scene.state().clone_host().unwrap();
    apply_impact(
        scene.model(),
        &mut state,
        &[impact.clone(), ImpactConstraint::Nonnegative(row)],
    )
    .unwrap();
    near(
        state.body_qd.host().unwrap()[id.index()].linear,
        Vec3::X * 0.5,
    );
    let mut state = scene.state().clone_host().unwrap();
    let before = state.clone_host().unwrap();
    let impossible = ImpactConstraint::Bilateral(ImpactRow {
        direct: None,
        ends: Vec::new(),
        velocity_bias: 1.,
    });
    let error = apply_impact(scene.model(), &mut state, &[impact, impossible]).unwrap_err();
    assert!(error.to_string().contains("did not converge"), "{error}");
    assert_eq!(
        state.joint_qd.host().unwrap(),
        before.joint_qd.host().unwrap()
    );
    assert_eq!(
        state.body_qd.host().unwrap(),
        before.body_qd.host().unwrap()
    );
    assert_eq!(state.body_q.host().unwrap(), before.body_q.host().unwrap());
}
