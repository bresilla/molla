use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxes, JointAxis, JointMotor, MotorModel};
use molla_sim::runtime::{BodyDesc, BodyHandle, JointDesc, JointHandle, JointSettings};
use molla_sim::{JointTargetMode, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

fn world() -> RigidWorld {
    let mut world = RigidWorld::new().unwrap();
    world.scene.set_gravity(Vec3::ZERO).unwrap();
    world
}

fn body(world: &mut RigidWorld, mass: f64) -> BodyHandle {
    world
        .scene
        .insert_body(BodyDesc {
            additional_mass: Some(MassProperties {
                mass,
                local_com: Vec3::ZERO,
                inertia: Mat3::from_diagonal(Vec3::splat(mass * 2.0)),
            }),
            ..BodyDesc::default()
        })
        .unwrap()
}

fn joint(
    world: &mut RigidWorld,
    parent: Option<BodyHandle>,
    child: BodyHandle,
    kind: JointType,
) -> JointHandle {
    world
        .scene
        .insert_joint(JointDesc {
            parent,
            child,
            kind,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
            axis: if kind == JointType::Prismatic {
                Vec3::Y
            } else {
                Vec3::Z
            },
        })
        .unwrap()
}

fn offset(world: &RigidWorld, joint: JointHandle) -> usize {
    let j = world.scene.joint_index(joint).unwrap().index();
    world.scene.model().joint_dof_offset.host().unwrap()[j] as usize
}

fn near(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-8, "{a} != {b}");
}

fn effort(force: f64, cap: f64) -> JointMotor {
    JointMotor {
        mode: JointTargetMode::Effort,
        target_position: force,
        max_force: cap,
        ..JointMotor::default()
    }
}

#[test]
fn semantic_axes_and_motor_caps_follow_authored_axes() {
    for (kind, axis) in [
        (JointType::Revolute, JointAxis::AngX),
        (JointType::Prismatic, JointAxis::LinX),
    ] {
        let mut world = world();
        let b = body(&mut world, 2.0);
        let j = joint(&mut world, None, b, kind);
        let stamp = world.scene.stamp();
        let revision = world.scene.revision();
        world
            .scene
            .set_joint_motor(j, axis, effort(100.0, 8.0))
            .unwrap();
        assert_eq!(world.scene.stamp(), stamp);
        assert_eq!(world.scene.revision(), revision);
        world.step(0.1).unwrap();
        let velocity = world.scene.body_velocity(b).unwrap();
        if kind == JointType::Revolute {
            near(velocity.angular.z, 0.2);
            near(velocity.angular.x, 0.0);
        } else {
            near(velocity.linear.y, 0.4);
            near(velocity.linear.x, 0.0);
        }
        assert!(
            world
                .scene
                .set_joint_motor(j, JointAxis::AngZ, effort(1.0, 1.0))
                .is_err()
        );
        assert!(!world.scene.joint_locked_axes(j).unwrap().contains(axis));
        assert_eq!(world.scene.joint_locked_axes(j).unwrap().0.count_ones(), 5);
    }
}

#[test]
fn motor_force_cap_does_not_cap_external_joint_force() {
    let mut world = world();
    let b = body(&mut world, 2.0);
    let j = joint(&mut world, None, b, JointType::Prismatic);
    world
        .scene
        .set_joint_motor(j, JointAxis::LinX, effort(100.0, 2.0))
        .unwrap();
    let dof = offset(&world, j);
    world.scene.control_mut().joint_force.host_mut().unwrap()[dof] = 8.0;
    world
        .scene
        .add_body_wrench(b, SpatialVector::new(Vec3::Y * 6.0, Vec3::ZERO))
        .unwrap();
    world.step(0.1).unwrap();
    near(world.scene.body_velocity(b).unwrap().linear.y, 0.8);
}

#[test]
fn acceleration_gains_are_mass_and_inertia_independent() {
    for (kind, axis) in [
        (JointType::Revolute, JointAxis::AngX),
        (JointType::Prismatic, JointAxis::LinX),
    ] {
        let mut velocities = Vec::new();
        for mass in [1.0, 1000.0] {
            let mut world = world();
            let b = body(&mut world, mass);
            let j = joint(&mut world, None, b, kind);
            world
                .scene
                .set_joint_motor(
                    j,
                    axis,
                    JointMotor {
                        mode: JointTargetMode::Velocity,
                        target_velocity: 2.0,
                        damping: 3.0,
                        model: MotorModel::Acceleration,
                        ..JointMotor::default()
                    },
                )
                .unwrap();
            world.step(0.1).unwrap();
            velocities.push(world.scene.state().joint_qd.host().unwrap()[offset(&world, j)]);
        }
        near(velocities[0], velocities[1]);
        near(
            velocities[0],
            2.0 * (1.0 - (1.0_f64 - 3.0 * 0.1 / 8.0).powi(8)),
        );
    }
}

#[test]
fn acceleration_motor_uses_floating_parent_effective_mass() {
    let mut world = world();
    let parent = body(&mut world, 3.0);
    let child = body(&mut world, 12.0);
    let j = joint(&mut world, Some(parent), child, JointType::Prismatic);
    world
        .scene
        .set_joint_motor(
            j,
            JointAxis::LinX,
            JointMotor {
                mode: JointTargetMode::Velocity,
                target_velocity: 2.0,
                damping: 3.0,
                model: MotorModel::Acceleration,
                ..JointMotor::default()
            },
        )
        .unwrap();
    world.step(0.1).unwrap();
    let relative = world.scene.state().joint_qd.host().unwrap()[offset(&world, j)];
    near(relative, 2.0 * (1.0 - (1.0_f64 - 3.0 * 0.1 / 8.0).powi(8)));
    let pv = world.scene.body_velocity(parent).unwrap().linear.y;
    let cv = world.scene.body_velocity(child).unwrap().linear.y;
    near(cv - pv, relative);
    near(3.0 * pv + 12.0 * cv, 0.0);
}

#[test]
fn motor_edits_survive_suspension_and_rebuilds() {
    let mut world = world();
    let p = body(&mut world, 1.0);
    let b = body(&mut world, 1.0);
    let j = joint(&mut world, Some(p), b, JointType::Revolute);
    world
        .scene
        .set_joint_motor(j, JointAxis::AngX, effort(3.0, 2.0))
        .unwrap();
    world.scene.set_body_enabled(p, false).unwrap();
    world
        .scene
        .set_joint_motor(j, JointAxis::AngX, effort(5.0, 4.0))
        .unwrap();
    assert_eq!(
        world.scene.joints_between(b, p).collect::<Vec<_>>(),
        vec![j]
    );
    world
        .scene
        .set_joint_settings(
            j,
            JointSettings {
                enabled: false,
                contacts_enabled: true,
            },
        )
        .unwrap();
    world.scene.set_body_enabled(p, true).unwrap();
    world
        .scene
        .set_joint_settings(j, JointSettings::default())
        .unwrap();
    let dof = offset(&world, j);
    assert_eq!(
        world.scene.control().joint_target_pos.host().unwrap()[dof],
        5.0
    );
    assert_eq!(
        world.scene.control().joint_motor_max_force.host().unwrap()[dof],
        4.0
    );
    world
        .scene
        .set_joint_limits(j, JointAxis::AngX, Some([-0.5, 0.5]))
        .unwrap();
    world.step(0.1).unwrap();
    assert!(world.scene.body_velocity(b).unwrap().angular.z > 0.0);
}

#[test]
fn invalid_axis_settings_are_atomic() {
    let mut world = world();
    let b = body(&mut world, 1.0);
    let j = joint(&mut world, None, b, JointType::D6);
    let stamp = world.scene.stamp();
    for limits in [[1.0, -1.0], [f64::NAN, 0.0], [0.0, f64::INFINITY]] {
        assert!(
            world
                .scene
                .set_joint_limits(j, JointAxis::LinY, Some(limits))
                .is_err()
        );
    }
    for bad in [
        JointMotor {
            stiffness: -1.0,
            ..JointMotor::default()
        },
        JointMotor {
            damping: f64::INFINITY,
            ..JointMotor::default()
        },
        effort(f64::NAN, 1.0),
        effort(1.0, f64::NAN),
        effort(1.0, -1.0),
    ] {
        assert!(
            world
                .scene
                .set_joint_motor(j, JointAxis::LinY, bad)
                .is_err()
        );
    }
    assert!(world.scene.set_joint_locked_axes(j, JointAxes(64)).is_err());
    assert_eq!(world.scene.stamp(), stamp);
    assert_eq!(
        world.scene.joint_axis_settings(j, JointAxis::LinY).unwrap(),
        Default::default()
    );
}

#[test]
fn generic_locks_limits_and_position_motor_drive_real_coordinates() {
    let mut world = world();
    let b = body(&mut world, 1.0);
    let j = joint(&mut world, None, b, JointType::D6);
    let locks = JointAxes::LIN.with(JointAxis::AngX).with(JointAxis::AngY);
    world.scene.set_joint_locked_axes(j, locks).unwrap();
    assert_eq!(world.scene.joint_locked_axes(j).unwrap(), locks);
    world
        .scene
        .set_joint_limits(j, JointAxis::AngZ, Some([-0.4, 0.4]))
        .unwrap();
    world
        .scene
        .set_joint_motor(
            j,
            JointAxis::AngZ,
            JointMotor {
                mode: JointTargetMode::Position,
                target_position: 0.25,
                stiffness: 40.0,
                damping: 15.0,
                model: MotorModel::Acceleration,
                ..JointMotor::default()
            },
        )
        .unwrap();
    world
        .scene
        .add_body_wrench(b, SpatialVector::new(Vec3::splat(10.0), Vec3::X * 10.0))
        .unwrap();
    for _ in 0..400 {
        world.step(0.01).unwrap();
    }
    let pose = world.scene.body_pose(b).unwrap();
    assert!(pose.position.length() < 1e-10);
    near(pose.rotation.x, 0.0);
    near(pose.rotation.y, 0.0);
    let angle = 2.0 * pose.rotation.z.atan2(pose.rotation.w);
    assert!((angle - 0.25).abs() < 1e-5, "{angle}");
    world
        .scene
        .set_joint_motor(j, JointAxis::AngZ, effort(10.0, f64::INFINITY))
        .unwrap();
    for _ in 0..100 {
        world.step(0.01).unwrap();
    }
    let pose = world.scene.body_pose(b).unwrap();
    near(2.0 * pose.rotation.z.atan2(pose.rotation.w), 0.4);
    near(world.scene.body_velocity(b).unwrap().angular.z, 0.0);
}

#[test]
fn ball_motor_holds_nonzero_angle_and_obeys_limits() {
    let mut world = world();
    let b = body(&mut world, 1.0);
    let j = joint(&mut world, None, b, JointType::Ball);
    assert_eq!(world.scene.joint_locked_axes(j).unwrap(), JointAxes::LIN);
    world
        .scene
        .set_joint_limits(j, JointAxis::AngY, Some([-0.3, 0.3]))
        .unwrap();
    world
        .scene
        .set_joint_motor(
            j,
            JointAxis::AngY,
            JointMotor {
                mode: JointTargetMode::Position,
                target_position: 0.2,
                stiffness: 40.0,
                damping: 15.0,
                model: MotorModel::Acceleration,
                ..JointMotor::default()
            },
        )
        .unwrap();
    for _ in 0..400 {
        world.step(0.01).unwrap();
    }
    let pose = world.scene.body_pose(b).unwrap();
    assert!((2.0 * pose.rotation.y.atan2(pose.rotation.w) - 0.2).abs() < 1e-5);
    world
        .scene
        .set_joint_motor(j, JointAxis::AngY, effort(-20.0, 20.0))
        .unwrap();
    for _ in 0..100 {
        world.step(0.01).unwrap();
    }
    let pose = world.scene.body_pose(b).unwrap();
    near(2.0 * pose.rotation.y.atan2(pose.rotation.w), -0.3);
    near(world.scene.body_velocity(b).unwrap().angular.y, 0.0);
}

#[test]
fn locking_a_moving_joint_removes_twist_and_can_be_released() {
    let mut world = world();
    let b = body(&mut world, 1.0);
    let j = joint(&mut world, None, b, JointType::D6);
    world
        .scene
        .set_body_velocity(
            b,
            SpatialVector::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.1, 0.2, 0.3)),
        )
        .unwrap();
    world.step(0.01).unwrap();
    world
        .scene
        .set_joint_locked_axes(j, JointAxes::ALL)
        .unwrap();
    world.step(0.01).unwrap();
    let pose = world.scene.body_pose(b).unwrap();
    assert!(pose.position.length() < 1e-12);
    assert!(pose.rotation.xyz().length() < 1e-12);
    assert_eq!(world.scene.body_velocity(b).unwrap(), SpatialVector::ZERO);
    world
        .scene
        .set_joint_limits(j, JointAxis::LinX, Some([-0.2, 0.2]))
        .unwrap();
    world
        .scene
        .set_joint_locked_axes(j, JointAxes::ALL)
        .unwrap();
    world
        .scene
        .set_joint_motor(j, JointAxis::LinX, effort(10.0, 10.0))
        .unwrap();
    world.step(0.1).unwrap();
    near(world.scene.body_pose(b).unwrap().position.x, 0.0);
    world
        .scene
        .set_joint_locked_axes(
            j,
            JointAxes::ANG.with(JointAxis::LinY).with(JointAxis::LinZ),
        )
        .unwrap();
    for _ in 0..20 {
        world.step(0.01).unwrap();
    }
    near(world.scene.body_pose(b).unwrap().position.x, 0.2);
    world
        .scene
        .set_joint_limits(j, JointAxis::LinX, None)
        .unwrap();
    world.step(0.1).unwrap();
    assert!(world.scene.body_pose(b).unwrap().position.x > 0.2);
}

#[test]
fn scalar_equality_limits_resist_effort_and_preserve_authored_motor() {
    let mut world = world();
    let b = body(&mut world, 1.0);
    let j = joint(&mut world, None, b, JointType::Revolute);
    world
        .scene
        .set_joint_motor(j, JointAxis::AngX, effort(10.0, 10.0))
        .unwrap();
    world
        .scene
        .set_joint_limits(j, JointAxis::AngX, Some([0.2, 0.2]))
        .unwrap();
    world.step(0.01).unwrap();
    let pose = world.scene.body_pose(b).unwrap();
    near(2.0 * pose.rotation.z.atan2(pose.rotation.w), 0.2);
    near(world.scene.body_velocity(b).unwrap().angular.z, 0.0);
    world
        .scene
        .set_joint_limits(j, JointAxis::AngX, None)
        .unwrap();
    world.step(0.01).unwrap();
    assert!(world.scene.body_velocity(b).unwrap().angular.z > 0.0);
}

#[test]
fn low_level_acceleration_controls_survive_dormancy_without_authoring() {
    let mut world = world();
    let b = body(&mut world, 1.0);
    let j = joint(&mut world, None, b, JointType::Prismatic);
    let dof = offset(&world, j);
    let control = world.scene.control_mut();
    control.joint_motor_model.host_mut().unwrap()[dof] = 1;
    control.joint_motor_max_force.host_mut().unwrap()[dof] = 0.5;
    control.joint_target_mode.host_mut().unwrap()[dof] = JointTargetMode::Velocity as u32;
    control.joint_target_vel.host_mut().unwrap()[dof] = 100.0;
    control.joint_target_kd.host_mut().unwrap()[dof] = 10.0;
    world.scene.set_body_enabled(b, false).unwrap();
    body(&mut world, 2.0);
    world.scene.set_body_enabled(b, true).unwrap();
    world.step(0.1).unwrap();
    near(world.scene.body_velocity(b).unwrap().linear.y, 0.05);
    assert_eq!(
        world
            .scene
            .joint_axis_settings(j, JointAxis::LinX)
            .unwrap()
            .motor,
        None
    );
}

#[test]
fn generic_translation_position_drive_holds_nonzero_target() {
    let mut world = world();
    let b = body(&mut world, 1.0);
    let j = joint(&mut world, None, b, JointType::D6);
    world
        .scene
        .set_joint_locked_axes(j, JointAxes::ANG)
        .unwrap();
    world
        .scene
        .set_joint_motor(
            j,
            JointAxis::LinY,
            JointMotor {
                mode: JointTargetMode::PositionVelocity,
                target_position: 0.7,
                stiffness: 40.0,
                damping: 15.0,
                ..JointMotor::default()
            },
        )
        .unwrap();
    for _ in 0..400 {
        world.step(0.01).unwrap();
    }
    assert!((world.scene.body_pose(b).unwrap().position.y - 0.7).abs() < 1e-5);
}
