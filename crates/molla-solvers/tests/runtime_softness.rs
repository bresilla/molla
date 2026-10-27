use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxes, JointAxis, JointMotor, JointSoftness};
use molla_sim::runtime::{BodyDesc, BodyHandle, BodyKind, JointDesc, JointHandle, JointSettings};
use molla_sim::{BodyParams, JointTargetMode, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

fn world() -> RigidWorld {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    w
}

fn body(w: &mut RigidWorld, position: Vec3) -> BodyHandle {
    w.scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_pose(Transform::from_translation(position)),
            additional_mass: Some(MassProperties {
                mass: 2.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * 4.0,
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
    axis: Vec3,
) -> JointHandle {
    w.scene
        .insert_joint(JointDesc {
            parent,
            child,
            kind,
            axis,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap()
}

fn soften(w: &mut RigidWorld, joint: JointHandle, hz: f64) {
    w.scene
        .set_joint_softness(
            joint,
            Some(JointSoftness {
                natural_frequency: hz,
                damping_ratio: 1.0,
            }),
        )
        .unwrap();
}

fn near(a: Vec3, b: Vec3, tolerance: f64) {
    assert!((a - b).length() < tolerance, "{a:?} != {b:?}");
}

fn dof(w: &RigidWorld, j: JointHandle) -> usize {
    w.scene.model().joint_dof_offset.host().unwrap()[w.scene.joint_index(j).unwrap().index()]
        as usize
}

#[test]
fn all_logical_kinds_lower_without_losing_handles_or_kinematics() {
    for kind in [
        JointType::Fixed,
        JointType::Ball,
        JointType::Revolute,
        JointType::Prismatic,
        JointType::D6,
    ] {
        let mut w = world();
        let b = body(&mut w, Vec3::new(0.2, -0.1, 0.3));
        let j = joint(&mut w, None, b, kind, Vec3::new(1.0, 2.0, -3.0));
        let before = w.scene.body_pose(b).unwrap();
        soften(&mut w, j, 8.0);
        assert_eq!(w.scene.joint(j).unwrap().kind, kind);
        assert_eq!(w.scene.joint(j).unwrap().child, b);
        assert_eq!(w.scene.body_pose(b).unwrap(), before);
        let index = w.scene.joint_index(j).unwrap().index();
        assert_eq!(
            w.scene.model().joint_type.host().unwrap()[index],
            JointType::D6 as u32
        );
        let mut state = w.scene.state().clone_host().unwrap();
        molla_sim::eval_fk(w.scene.model(), &mut state).unwrap();
        let reconstructed = state.body_q.host().unwrap()[w.scene.body_index(b).unwrap().index()];
        near(reconstructed.position, before.position, 1e-10);
        assert!(reconstructed.rotation.angle_between(before.rotation) < 1e-7);
        let active = w
            .scene
            .model()
            .joint_compliance
            .host()
            .unwrap()
            .iter()
            .filter(|c| c.enabled != 0)
            .count();
        assert_eq!(
            active,
            match kind {
                JointType::Fixed => 6,
                JointType::Ball => 3,
                JointType::D6 => 0,
                _ => 5,
            }
        );
    }
}

#[test]
fn frame_capture_moves_by_force_and_settles_at_new_rest_frame() {
    let mut w = world();
    let b = body(&mut w, Vec3::X * 2.0);
    let j = w
        .scene
        .insert_joint(JointDesc {
            parent: None,
            child: b,
            kind: JointType::D6,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::from_translation(-Vec3::X * 2.0),
            axis: Vec3::X,
        })
        .unwrap();
    w.scene.set_joint_locked_axes(j, JointAxes::ALL).unwrap();
    soften(&mut w, j, 8.0);
    for tick in 1..=120 {
        let before = w.scene.body_pose(b).unwrap();
        let velocity = w.scene.body_velocity(b).unwrap();
        let t = tick as f64 / 120.0;
        let s = t * t * (3.0 - 2.0 * t);
        w.scene
            .set_joint_frames(
                j,
                Transform::IDENTITY,
                Transform::from_translation(-Vec3::X * 2.0 * (1.0 - s)),
            )
            .unwrap();
        assert_eq!(w.scene.body_pose(b).unwrap(), before);
        assert_eq!(w.scene.body_velocity(b).unwrap(), velocity);
        let stamp = w.scene.stamp();
        soften(&mut w, j, 8.0 + 22.0 * s);
        assert_eq!(w.scene.stamp(), stamp);
        w.step(1.0 / 120.0).unwrap();
        assert!(w.quarantined_bodies().is_empty());
        assert!((w.scene.body_pose(b).unwrap().position - before.position).length() < 0.1);
    }
    for _ in 0..60 {
        w.step(1.0 / 120.0).unwrap();
    }
    near(w.scene.body_pose(b).unwrap().position, Vec3::ZERO, 1e-6);
    near(w.scene.body_velocity(b).unwrap().linear, Vec3::ZERO, 1e-6);
}

#[test]
fn semantic_motors_follow_scalar_axes_and_ball_slots() {
    for kind in [JointType::Revolute, JointType::Prismatic, JointType::Ball] {
        let mut w = world();
        let b = body(&mut w, Vec3::ZERO);
        let axis = Vec3::new(1.0, 2.0, -3.0).normalize();
        let j = joint(&mut w, None, b, kind, axis);
        soften(&mut w, j, 8.0);
        let semantic = match kind {
            JointType::Prismatic => JointAxis::LinX,
            JointType::Ball => JointAxis::AngZ,
            _ => JointAxis::AngX,
        };
        w.scene
            .set_joint_motor(
                j,
                semantic,
                JointMotor {
                    mode: JointTargetMode::Effort,
                    target_position: 100.0,
                    max_force: 8.0,
                    ..Default::default()
                },
            )
            .unwrap();
        w.step(0.1).unwrap();
        let velocity = w.scene.body_velocity(b).unwrap();
        if kind == JointType::Prismatic {
            near(velocity.linear, axis * 0.4, 1e-8);
        } else {
            near(
                velocity.angular,
                if kind == JointType::Ball {
                    Vec3::Z
                } else {
                    axis
                } * 0.2,
                1e-8,
            );
        }
    }
}

#[test]
fn raw_controls_survive_layout_toggles_and_suspension() {
    let mut w = world();
    let b = body(&mut w, Vec3::ZERO);
    let j = joint(&mut w, None, b, JointType::Revolute, Vec3::Y);
    let offset = dof(&w, j);
    w.scene.control_mut().joint_force.host_mut().unwrap()[offset] = 7.0;
    w.scene.control_mut().joint_target_vel.host_mut().unwrap()[offset] = 2.0;
    soften(&mut w, j, 8.0);
    let offset = dof(&w, j) + 3;
    assert_eq!(w.scene.control().joint_force.host().unwrap()[offset], 7.0);
    assert_eq!(
        w.scene.control().joint_target_vel.host().unwrap()[offset],
        2.0
    );
    w.scene
        .set_joint_settings(
            j,
            JointSettings {
                enabled: false,
                ..Default::default()
            },
        )
        .unwrap();
    w.scene.set_joint_softness(j, None).unwrap();
    w.scene
        .set_joint_settings(j, JointSettings::default())
        .unwrap();
    let offset = dof(&w, j);
    assert_eq!(w.scene.control().joint_force.host().unwrap()[offset], 7.0);
    assert_eq!(
        w.scene.control().joint_target_vel.host().unwrap()[offset],
        2.0
    );
    w.step(0.1).unwrap();
    near(
        w.scene.body_velocity(b).unwrap().angular,
        Vec3::Y * 0.175,
        1e-8,
    );
}

#[test]
fn reversed_soft_fixed_joint_recovers_without_pose_jump() {
    let mut w = world();
    let a = body(&mut w, Vec3::X * 0.3);
    let b = body(&mut w, Vec3::ZERO);
    w.scene.set_body_kind(b, BodyKind::Fixed).unwrap();
    let j = joint(&mut w, Some(a), b, JointType::Fixed, Vec3::X);
    soften(&mut w, j, 8.0);
    let index = w.scene.joint_index(j).unwrap().index();
    assert_eq!(w.scene.model().joint_reversed.host().unwrap()[index], 1);
    near(w.scene.body_pose(a).unwrap().position, Vec3::X * 0.3, 1e-10);
    for _ in 0..120 {
        w.step(1.0 / 120.0).unwrap();
    }
    near(w.scene.body_pose(a).unwrap().position, Vec3::ZERO, 1e-7);
    near(w.scene.body_pose(b).unwrap().position, Vec3::ZERO, 1e-10);
}

#[test]
fn invalid_softness_and_frames_are_atomic_even_when_suspended() {
    let mut w = world();
    let b = body(&mut w, Vec3::ZERO);
    let j = joint(&mut w, None, b, JointType::Fixed, Vec3::X);
    soften(&mut w, j, 8.0);
    for enabled in [true, false] {
        w.scene
            .set_joint_settings(
                j,
                JointSettings {
                    enabled,
                    ..Default::default()
                },
            )
            .unwrap();
        let stamp = w.scene.stamp();
        let before = w.scene.joint_softness(j).unwrap();
        assert!(
            w.scene
                .set_joint_softness(
                    j,
                    Some(JointSoftness {
                        natural_frequency: f64::NAN,
                        damping_ratio: 1.0
                    })
                )
                .is_err()
        );
        assert!(
            w.scene
                .set_joint_frames(
                    j,
                    Transform {
                        rotation: Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
                        ..Transform::IDENTITY
                    },
                    Transform::IDENTITY
                )
                .is_err()
        );
        assert_eq!(w.scene.stamp(), stamp);
        assert_eq!(w.scene.joint_softness(j).unwrap(), before);
    }
}

#[test]
fn soft_velocity_and_wrenches_survive_unrelated_rebuilds() {
    let mut w = world();
    let b = body(&mut w, Vec3::ZERO);
    let j = joint(&mut w, None, b, JointType::Ball, Vec3::X);
    soften(&mut w, j, 0.0);
    let velocity = SpatialVector::new(Vec3::new(0.1, -0.2, 0.3), Vec3::new(-0.2, 0.3, 0.1));
    w.scene.set_body_velocity(b, velocity).unwrap();
    w.scene
        .add_body_wrench(b, SpatialVector::new(Vec3::X, Vec3::Y))
        .unwrap();
    w.step(0.01).unwrap();
    let pose = w.scene.body_pose(b).unwrap();
    let velocity = w.scene.body_velocity(b).unwrap();
    body(&mut w, Vec3::splat(100.0));
    assert_eq!(w.scene.body_pose(b).unwrap(), pose);
    near(
        w.scene.body_velocity(b).unwrap().linear,
        velocity.linear,
        1e-10,
    );
    near(
        w.scene.body_velocity(b).unwrap().angular,
        velocity.angular,
        1e-10,
    );
    let mut state = w.scene.state().clone_host().unwrap();
    molla_sim::eval_fk(w.scene.model(), &mut state).unwrap();
    molla_sim::eval_body_velocities(w.scene.model(), &mut state).unwrap();
    let index = w.scene.body_index(b).unwrap().index();
    near(
        state.body_qd.host().unwrap()[index].linear,
        velocity.linear,
        1e-10,
    );
    near(
        state.body_qd.host().unwrap()[index].angular,
        velocity.angular,
        1e-10,
    );
}

#[test]
fn soft_locks_and_limits_change_rest_bounds_without_projection() {
    let mut w = world();
    let b = body(&mut w, Vec3::X * 0.4);
    let j = joint(&mut w, None, b, JointType::D6, Vec3::X);
    soften(&mut w, j, 8.0);
    w.scene
        .set_joint_limits(j, JointAxis::LinX, Some([-0.1, 0.1]))
        .unwrap();
    near(w.scene.body_pose(b).unwrap().position, Vec3::X * 0.4, 1e-10);
    w.step(1.0 / 120.0).unwrap();
    assert!(w.scene.body_pose(b).unwrap().position.x > 0.1);
    let pose = w.scene.body_pose(b).unwrap();
    w.scene.set_joint_locked_axes(j, JointAxes::LIN).unwrap();
    assert_eq!(w.scene.body_pose(b).unwrap(), pose);
    let offset = dof(&w, j);
    assert_eq!(
        w.scene.model().joint_compliance.host().unwrap()[offset].lower,
        0.0
    );
    for _ in 0..120 {
        w.step(1.0 / 120.0).unwrap();
    }
    near(w.scene.body_pose(b).unwrap().position, Vec3::ZERO, 1e-7);
}

#[test]
fn authored_motors_survive_softness_toggles_and_reversed_axis_alignment() {
    for kind in [JointType::Revolute, JointType::Prismatic] {
        let mut w = world();
        let a = body(&mut w, Vec3::ZERO);
        let b = body(&mut w, Vec3::ZERO);
        w.scene.set_body_kind(b, BodyKind::Fixed).unwrap();
        let axis = Vec3::new(1.0, -2.0, 3.0).normalize();
        let j = joint(&mut w, Some(a), b, kind, axis);
        let semantic = if kind == JointType::Revolute {
            JointAxis::AngX
        } else {
            JointAxis::LinX
        };
        let motor = JointMotor {
            mode: JointTargetMode::Effort,
            target_position: 8.0,
            max_force: 4.0,
            ..Default::default()
        };
        w.scene.set_joint_motor(j, semantic, motor).unwrap();
        soften(&mut w, j, 8.0);
        w.scene.set_joint_softness(j, None).unwrap();
        soften(&mut w, j, 30.0);
        assert_eq!(
            w.scene.joint_axis_settings(j, semantic).unwrap().motor,
            Some(motor)
        );
        w.step(0.1).unwrap();
        let velocity = w.scene.body_velocity(a).unwrap();
        if kind == JointType::Revolute {
            near(velocity.angular, -axis * 0.1, 1e-8);
        } else {
            near(velocity.linear, -axis * 0.2, 1e-8);
        }
        assert!(w.quarantined_bodies().is_empty());
    }
}

#[test]
fn suspended_softness_updates_survive_resume_and_body_removal() {
    let mut w = world();
    let b = body(&mut w, Vec3::ZERO);
    let j = joint(&mut w, None, b, JointType::Fixed, Vec3::X);
    soften(&mut w, j, 8.0);
    w.scene
        .set_joint_settings(
            j,
            JointSettings {
                enabled: false,
                ..Default::default()
            },
        )
        .unwrap();
    soften(&mut w, j, 30.0);
    w.scene
        .set_joint_frames(j, Transform::from_translation(Vec3::X), Transform::IDENTITY)
        .unwrap();
    w.scene
        .set_joint_settings(j, JointSettings::default())
        .unwrap();
    near(w.scene.body_pose(b).unwrap().position, Vec3::ZERO, 1e-10);
    assert!(
        w.scene
            .model()
            .joint_compliance
            .host()
            .unwrap()
            .iter()
            .all(|c| c.natural_frequency == 30.0)
    );
    for _ in 0..120 {
        w.step(1.0 / 120.0).unwrap();
    }
    near(w.scene.body_pose(b).unwrap().position, Vec3::X, 1e-7);
    w.scene.remove_body(b).unwrap();
    assert!(w.scene.joint_softness(j).is_err());
    assert_eq!(w.scene.model().joint_dof_total, 0);
}
