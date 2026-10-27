use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxis, JointMotor, MotorModel};
use molla_sim::runtime::{BodyDesc, BodyHandle, BodyKind, JointDesc, JointHandle};
use molla_sim::{JointTargetMode, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

fn world() -> RigidWorld {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    w
}

fn body(w: &mut RigidWorld, position: Vec3) -> BodyHandle {
    w.scene
        .insert_body(BodyDesc {
            params: molla_sim::BodyParams {
                pose: Transform::from_translation(position),
                ..Default::default()
            },
            additional_mass: Some(MassProperties {
                mass: 2.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::from_diagonal(Vec3::splat(4.0)),
            }),
            ..Default::default()
        })
        .unwrap()
}

fn joint(
    w: &mut RigidWorld,
    parent: Option<BodyHandle>,
    child: BodyHandle,
    kind: JointType,
) -> JointHandle {
    let xp = parent
        .map(|b| w.scene.body_pose(b).unwrap())
        .unwrap_or(Transform::IDENTITY);
    let xc = w.scene.body_pose(child).unwrap();
    w.scene
        .insert_joint(JointDesc {
            parent,
            child,
            kind,
            frame_parent: Transform::IDENTITY,
            frame_child: xc.inverse().compose(&xp),
            axis: Vec3::Z,
        })
        .unwrap()
}

fn reversed(w: &RigidWorld, j: JointHandle) -> bool {
    w.scene.model().joint_reversed.host().unwrap()[w.scene.joint_index(j).unwrap().index()] != 0
}

fn coordinate(w: &RigidWorld, j: JointHandle) -> f64 {
    let offset = w.scene.model().joint_coord_offset.host().unwrap()
        [w.scene.joint_index(j).unwrap().index()] as usize;
    w.scene.state().joint_q.host().unwrap()[offset]
}

fn near(a: Vec3, b: Vec3, tolerance: f64) {
    assert!((a - b).length() < tolerance, "{a:?} != {b:?}");
}

#[test]
fn fixing_a_hinge_child_reroots_without_reversing_motor_semantics() {
    let mut w = world();
    let a = body(&mut w, Vec3::ZERO);
    let b = body(&mut w, Vec3::ZERO);
    let j = joint(&mut w, Some(a), b, JointType::Revolute);
    w.scene
        .set_joint_motor(
            j,
            JointAxis::AngX,
            JointMotor {
                mode: JointTargetMode::Effort,
                target_position: 8.0,
                ..Default::default()
            },
        )
        .unwrap();
    w.scene.set_body_kind(b, BodyKind::Fixed).unwrap();
    assert!(reversed(&w, j));
    assert_eq!(w.scene.joint(j).unwrap().parent, Some(a));
    assert_eq!(w.scene.joint(j).unwrap().child, b);
    w.step(0.1).unwrap();
    near(
        w.scene.body_velocity(a).unwrap().angular,
        Vec3::Z * -0.2,
        1e-10,
    );
    assert_eq!(w.scene.body_pose(b).unwrap(), Transform::IDENTITY);
    assert!(coordinate(&w, j) > 0.0);
    let q = coordinate(&w, j);
    let va = w.scene.body_velocity(a).unwrap();
    w.scene.set_body_kind(b, BodyKind::Dynamic).unwrap();
    assert!(!reversed(&w, j));
    assert_eq!(coordinate(&w, j), q);
    assert_eq!(w.scene.body_velocity(a).unwrap(), va);
    w.step(0.1).unwrap();
    assert!(w.scene.body_velocity(b).unwrap().angular.z > 0.0);
}

#[test]
fn world_anchor_can_pin_a_nonroot_and_detach_with_current_velocity() {
    let mut w = world();
    let a = body(&mut w, Vec3::X);
    let b = body(&mut w, Vec3::ZERO);
    let j = joint(&mut w, Some(a), b, JointType::Revolute);
    let pin = joint(&mut w, None, b, JointType::Fixed);
    assert!(reversed(&w, j));
    w.apply_body_impulse(a, SpatialVector::new(Vec3::ZERO, Vec3::Z * 4.0))
        .unwrap();
    near(w.scene.body_velocity(a).unwrap().angular, Vec3::Z, 1e-10);
    w.step(0.01).unwrap();
    let pose = w.scene.body_pose(a).unwrap();
    let velocity = w.scene.body_velocity(a).unwrap();
    w.scene.remove_joint(pin).unwrap();
    assert_eq!(w.scene.body_pose(a).unwrap(), pose);
    assert_eq!(w.scene.body_velocity(a).unwrap(), velocity);
    assert!(!reversed(&w, j));
    w.step(0.01).unwrap();
    assert!(w.scene.body_velocity(a).unwrap().angular.is_finite());
}

#[test]
fn merging_at_two_existing_children_reverses_only_the_needed_branch() {
    let mut w = world();
    let a = body(&mut w, Vec3::ZERO);
    let b = body(&mut w, Vec3::ZERO);
    let c = body(&mut w, Vec3::ZERO);
    let d = body(&mut w, Vec3::ZERO);
    let ab = joint(&mut w, Some(a), b, JointType::Fixed);
    let cd = joint(&mut w, Some(c), d, JointType::Revolute);
    for body in [a, b, c, d] {
        w.scene
            .set_body_velocity(body, SpatialVector::new(Vec3::X, Vec3::ZERO))
            .unwrap();
    }
    let hitch = joint(&mut w, Some(b), d, JointType::Fixed);
    assert!(!reversed(&w, ab));
    assert!(reversed(&w, cd));
    assert_eq!(w.scene.model().articulation_count, 1);
    w.scene
        .set_joint_motor(
            cd,
            JointAxis::AngX,
            JointMotor {
                mode: JointTargetMode::Effort,
                target_position: 4.0,
                ..Default::default()
            },
        )
        .unwrap();
    w.step(0.1).unwrap();
    for body in [a, b, c, d] {
        near(
            w.scene.body_pose(body).unwrap().position,
            Vec3::X * 0.1,
            1e-10,
        );
    }
    assert!(w.scene.body_velocity(c).unwrap().angular.z < 0.0);
    assert!(w.scene.body_velocity(d).unwrap().angular.z > 0.0);
    let before: Vec<_> = [a, b, c, d]
        .iter()
        .map(|&b| {
            (
                w.scene.body_pose(b).unwrap(),
                w.scene.body_velocity(b).unwrap(),
            )
        })
        .collect();
    w.scene.remove_joint(hitch).unwrap();
    assert_eq!(w.scene.model().articulation_count, 2);
    assert!(!reversed(&w, cd));
    for (i, body) in [a, b, c, d].iter().enumerate() {
        assert_eq!(w.scene.body_pose(*body).unwrap(), before[i].0);
        assert_eq!(w.scene.body_velocity(*body).unwrap(), before[i].1);
    }
    w.step(0.01).unwrap();
    for body in [a, b, c, d] {
        assert!(w.scene.body_pose(body).unwrap().position.is_finite());
    }
}

#[test]
fn reversed_prismatic_limits_and_impulses_keep_logical_signs() {
    for origin in [Vec3::ZERO, Vec3::new(1e6, -2e6, 3e6)] {
        let mut w = world();
        let a = body(&mut w, origin);
        let b = body(&mut w, origin);
        let j = joint(&mut w, Some(a), b, JointType::Prismatic);
        w.scene.set_body_kind(b, BodyKind::Fixed).unwrap();
        w.apply_body_impulse(a, SpatialVector::new(Vec3::Z * -2.0, Vec3::ZERO))
            .unwrap();
        near(w.scene.body_velocity(a).unwrap().linear, -Vec3::Z, 1e-8);
        w.scene
            .set_joint_limits(j, JointAxis::LinX, Some([-0.1, 0.1]))
            .unwrap();
        w.scene
            .set_joint_motor(
                j,
                JointAxis::LinX,
                JointMotor {
                    mode: JointTargetMode::Effort,
                    target_position: 20.0,
                    ..Default::default()
                },
            )
            .unwrap();
        w.step(0.2).unwrap();
        near(
            w.scene.body_pose(a).unwrap().position,
            origin - Vec3::Z * 0.1,
            1e-8,
        );
        near(w.scene.body_velocity(a).unwrap().linear, Vec3::ZERO, 1e-10);
        assert!((coordinate(&w, j) - 0.1).abs() < 1e-10);
    }
}

#[test]
fn reversed_ball_motor_targets_authored_relative_orientation() {
    let mut w = world();
    let a = body(&mut w, Vec3::ZERO);
    let b = body(&mut w, Vec3::ZERO);
    let j = joint(&mut w, Some(a), b, JointType::Ball);
    w.scene.set_body_kind(b, BodyKind::Fixed).unwrap();
    w.scene
        .set_joint_motor(
            j,
            JointAxis::AngY,
            JointMotor {
                mode: JointTargetMode::Position,
                target_position: 0.3,
                stiffness: 40.0,
                damping: 15.0,
                model: MotorModel::Acceleration,
                ..Default::default()
            },
        )
        .unwrap();
    for _ in 0..400 {
        w.step(0.01).unwrap();
    }
    let pose = w.scene.body_pose(a).unwrap();
    assert!((2.0 * pose.rotation.y.atan2(pose.rotation.w) + 0.3).abs() < 1e-5);
    assert_eq!(w.scene.body_pose(b).unwrap(), Transform::IDENTITY);
}

#[test]
fn reversed_free_joint_matches_unconstrained_body_motion() {
    let mut w = world();
    let a = body(&mut w, Vec3::new(0.4, 0.6, -0.2));
    let b = body(&mut w, Vec3::new(-0.2, 0.3, 0.5));
    w.scene
        .set_body_pose(
            a,
            Transform::new(Vec3::new(0.4, 0.6, -0.2), Quat::from_rotation_y(0.3)),
        )
        .unwrap();
    let initial = SpatialVector::new(Vec3::new(0.3, 0.4, 0.5), Vec3::new(0.2, 0.4, 0.6));
    let pose = w.scene.body_pose(a).unwrap();
    w.scene.set_body_velocity(a, initial).unwrap();
    w.scene.set_body_kind(b, BodyKind::Fixed).unwrap();
    let j = w
        .scene
        .insert_joint(JointDesc {
            parent: Some(a),
            child: b,
            kind: JointType::D6,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
            axis: Vec3::X,
        })
        .unwrap();
    assert!(reversed(&w, j));
    w.step(1e-5).unwrap();
    let actual = w.scene.body_velocity(a).unwrap();
    near(actual.linear, initial.linear, 1e-9);
    near(actual.angular, initial.angular, 1e-9);
    let after = w.scene.body_pose(a).unwrap();
    near(after.position, pose.position + initial.linear * 1e-5, 1e-9);
    let rotation = Quat::from_scaled_axis(initial.angular * 1e-5) * pose.rotation;
    assert!((after.rotation.dot(rotation).abs() - 1.0).abs() < 1e-12);
}

#[test]
fn rerooted_teleport_moves_solver_descendants_not_authored_children() {
    let mut w = world();
    let a = body(&mut w, Vec3::X);
    let b = body(&mut w, Vec3::ZERO);
    let j = joint(&mut w, Some(a), b, JointType::Fixed);
    w.scene.set_body_kind(b, BodyKind::Kinematic).unwrap();
    assert!(reversed(&w, j));
    w.scene
        .set_body_pose(b, Transform::new(Vec3::Y * 2.0, Quat::from_rotation_z(0.4)))
        .unwrap();
    near(
        w.scene.body_pose(a).unwrap().position,
        Vec3::Y * 2.0 + Quat::from_rotation_z(0.4) * Vec3::X,
        1e-10,
    );
    w.scene
        .set_body_velocity(b, SpatialVector::new(Vec3::Y, Vec3::Z))
        .unwrap();
    let expected = Vec3::Y
        + Vec3::Z
            .cross(w.scene.body_pose(a).unwrap().position - w.scene.body_pose(b).unwrap().position);
    near(w.scene.body_velocity(a).unwrap().linear, expected, 1e-10);
}

#[test]
fn constraint_loops_remain_atomic_errors() {
    let mut w = world();
    let a = body(&mut w, Vec3::ZERO);
    let b = body(&mut w, Vec3::ZERO);
    joint(&mut w, Some(a), b, JointType::Fixed);
    w.scene.set_body_kind(b, BodyKind::Fixed).unwrap();
    let stamp = w.scene.stamp();
    assert!(w.scene.set_body_kind(a, BodyKind::Fixed).is_err());
    assert_eq!(w.scene.body(a).unwrap().kind, BodyKind::Dynamic);
    assert_eq!(w.scene.stamp(), stamp);
    assert!(
        w.scene
            .insert_joint(JointDesc {
                parent: Some(b),
                child: a,
                kind: JointType::Fixed,
                frame_parent: Transform::IDENTITY,
                frame_child: Transform::IDENTITY,
                axis: Vec3::X
            })
            .is_err()
    );
    assert_eq!(w.scene.stamp(), stamp);
}

fn advance_d6(model: &molla_sim::Model, state: &molla_sim::State, dt: f64) -> molla_sim::State {
    let mut next = state.clone_host().unwrap();
    let qd = state.joint_qd.host().unwrap();
    let q = next.joint_q.host_mut().unwrap();
    for k in 0..3 {
        q[k] += qd[k] * dt;
    }
    let rotation = Quat::from_xyzw(q[3], q[4], q[5], q[6]);
    let delta = Quat::from_scaled_axis(Vec3::new(qd[3], qd[4], qd[5]) * dt);
    q[3..7].copy_from_slice(&(delta * rotation).to_array());
    molla_sim::eval_fk(model, &mut next).unwrap();
    molla_sim::eval_body_velocities(model, &mut next).unwrap();
    next
}

#[test]
fn reversed_d6_jacobian_and_inverse_dynamics_match_finite_differences() {
    let mut w = world();
    let a = body(&mut w, Vec3::new(0.4, -0.2, 0.7));
    let b = body(&mut w, Vec3::new(-0.5, 0.8, 0.3));
    w.scene
        .set_body_pose(
            a,
            Transform::new(Vec3::new(0.4, -0.2, 0.7), Quat::from_rotation_y(0.6)),
        )
        .unwrap();
    w.scene.set_body_kind(b, BodyKind::Fixed).unwrap();
    let inertia_rotation = Mat3::from_quat(Quat::from_rotation_z(0.3));
    let mass = MassProperties {
        mass: 3.0,
        local_com: Vec3::new(0.2, 0.3, -0.1),
        inertia: inertia_rotation
            * Mat3::from_diagonal(Vec3::new(2.0, 3.0, 4.0))
            * inertia_rotation.transpose(),
    };
    w.set_body_additional_mass(a, mass).unwrap();
    w.scene
        .insert_joint(JointDesc {
            parent: Some(a),
            child: b,
            kind: JointType::D6,
            frame_parent: Transform::new(Vec3::new(0.2, -0.1, 0.3), Quat::from_rotation_x(0.2)),
            frame_child: Transform::new(Vec3::new(-0.1, 0.3, 0.2), Quat::from_rotation_z(-0.4)),
            axis: Vec3::X,
        })
        .unwrap();
    let model = w.scene.model();
    let index = w.scene.body_index(a).unwrap().index();
    let mut state = w.scene.state().clone_host().unwrap();
    molla_sim::eval_fk(model, &mut state).unwrap();
    let jac = molla_solvers::jacobian(model, &state, 0, index, mass.local_com).unwrap();
    let epsilon = 1e-6;
    for k in 0..6 {
        state.joint_qd.host_mut().unwrap().fill(0.0);
        state.joint_qd.host_mut().unwrap()[k] = 1.0;
        let before = advance_d6(model, &state, -epsilon);
        let after = advance_d6(model, &state, epsilon);
        let bp = before.body_q.host().unwrap()[index];
        let ap = after.body_q.host().unwrap()[index];
        let linear = (ap.position + ap.rotation * mass.local_com
            - bp.position
            - bp.rotation * mass.local_com)
            / (2.0 * epsilon);
        let angular = (ap.rotation * bp.rotation.conjugate()).to_scaled_axis() / (2.0 * epsilon);
        near(Vec3::new(jac[k], jac[6 + k], jac[12 + k]), linear, 1e-8);
        near(
            Vec3::new(jac[18 + k], jac[24 + k], jac[30 + k]),
            angular,
            1e-8,
        );
    }
    state
        .joint_qd
        .host_mut()
        .unwrap()
        .copy_from_slice(&[0.3, -0.2, 0.4, 0.5, 0.7, -0.6]);
    molla_sim::eval_body_velocities(model, &mut state).unwrap();
    let before = advance_d6(model, &state, -epsilon);
    let after = advance_d6(model, &state, epsilon);
    let vb = before.body_qd.host().unwrap()[index];
    let va = after.body_qd.host().unwrap()[index];
    let linear_acceleration = (va.linear - vb.linear) / (2.0 * epsilon);
    let angular_acceleration = (va.angular - vb.angular) / (2.0 * epsilon);
    let rotation = Mat3::from_quat(state.body_q.host().unwrap()[index].rotation);
    let inertia = rotation * mass.inertia * rotation.transpose();
    let omega = state.body_qd.host().unwrap()[index].angular;
    let force = linear_acceleration * mass.mass;
    let torque = inertia * angular_acceleration + omega.cross(inertia * omega);
    let rnea = molla_solvers::inverse_dynamics(model, &state, 0, &[0.0; 6]).unwrap();
    for k in 0..6 {
        let expected = Vec3::new(jac[k], jac[6 + k], jac[12 + k]).dot(force)
            + Vec3::new(jac[18 + k], jac[24 + k], jac[30 + k]).dot(torque);
        assert!(
            (rnea[k] - expected).abs() < 1e-8,
            "axis {k}: {} != {expected}",
            rnea[k]
        );
    }
    let matrix = molla_solvers::mass_matrix(model, &state, 0).unwrap();
    for row in 0..6 {
        for col in 0..6 {
            let linear = |i| Vec3::new(jac[i], jac[6 + i], jac[12 + i]);
            let angular = |i| Vec3::new(jac[18 + i], jac[24 + i], jac[30 + i]);
            let expected =
                mass.mass * linear(row).dot(linear(col)) + angular(row).dot(inertia * angular(col));
            assert!((matrix[row * 6 + col] - expected).abs() < 1e-9);
        }
    }
    let initial_qd = state.joint_qd.host().unwrap().to_vec();
    w.scene.commit_step(state).unwrap();
    w.scene
        .control_mut()
        .joint_force
        .host_mut()
        .unwrap()
        .copy_from_slice(&rnea);
    w.step(1e-5).unwrap();
    for (&actual, &expected) in w
        .scene
        .state()
        .joint_qd
        .host()
        .unwrap()
        .iter()
        .zip(&initial_qd)
    {
        assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
    }
}

#[test]
fn reversed_frame_edits_and_suspension_keep_authored_endpoints() {
    let mut w = world();
    let a = body(&mut w, Vec3::X);
    let b = body(&mut w, Vec3::ZERO);
    let j = joint(&mut w, Some(a), b, JointType::Fixed);
    w.scene.set_body_kind(b, BodyKind::Fixed).unwrap();
    w.scene
        .set_joint_frames(
            j,
            Transform::from_translation(Vec3::X * 2.0),
            Transform::from_translation(Vec3::X),
        )
        .unwrap();
    w.step(0.01).unwrap();
    near(w.scene.body_pose(a).unwrap().position, -Vec3::X, 1e-10);
    w.scene.set_body_enabled(b, false).unwrap();
    assert!(w.scene.joint_index(j).is_none());
    w.scene
        .set_body_velocity(a, SpatialVector::new(Vec3::Y, Vec3::ZERO))
        .unwrap();
    w.step(0.1).unwrap();
    near(
        w.scene.body_pose(a).unwrap().position,
        -Vec3::X + Vec3::Y * 0.1,
        1e-10,
    );
    w.scene.set_body_enabled(b, true).unwrap();
    assert!(reversed(&w, j));
    w.step(0.01).unwrap();
    near(w.scene.body_pose(a).unwrap().position, -Vec3::X, 1e-10);
    assert_eq!(w.scene.joints_between(a, b).collect::<Vec<_>>(), vec![j]);
}
