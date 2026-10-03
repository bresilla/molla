use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxis, JointMotor, JointSoftness};
use molla_sim::runtime::{BodyDesc, BodyHandle, BodyKind, JointDesc, JointSettings};
use molla_sim::{BodyParams, JointTargetMode, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

fn body(world: &mut RigidWorld, position: Vec3, mass: f64) -> BodyHandle {
    world
        .scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_pose(Transform::from_translation(position)),
            additional_mass: Some(MassProperties {
                mass,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * mass,
            }),
            ..Default::default()
        })
        .unwrap()
}

fn desc(parent: Option<BodyHandle>, child: BodyHandle, kind: JointType) -> JointDesc {
    JointDesc {
        parent,
        child,
        kind,
        axis: Vec3::X,
        frame_parent: Transform::IDENTITY,
        frame_child: Transform::IDENTITY,
    }
}

fn softness() -> JointSoftness {
    JointSoftness {
        natural_frequency: 30.0,
        damping_ratio: 1.0,
    }
}

#[test]
fn loop_joint_does_not_project_and_survives_tree_edits() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    let a = body(&mut w, Vec3::ZERO, 1000.0);
    let b = body(&mut w, Vec3::X, 1.0);
    let before = w.scene.body_pose(b).unwrap();
    let j = w
        .scene
        .insert_loop_joint(desc(Some(a), b, JointType::Ball), softness())
        .unwrap();
    assert_eq!(w.scene.body_pose(b).unwrap(), before);
    assert!(w.scene.joint_index(j).is_none());
    w.scene.set_joint_softness(j, None).unwrap();
    assert_eq!(w.scene.body_pose(b).unwrap(), before);
    assert_eq!(w.scene.rigid_ball_joints().count(), 1);
    w.scene.set_joint_softness(j, Some(softness())).unwrap();
    assert_eq!(w.scene.rigid_ball_joints().count(), 0);
    for _ in 0..120 {
        w.step(1.0 / 120.0).unwrap();
    }
    assert!(
        (w.scene.body_pose(a).unwrap().position - w.scene.body_pose(b).unwrap().position).length()
            < 1e-7
    );
    let momentum = w.scene.body_velocity(a).unwrap().linear * 1000.0
        + w.scene.body_velocity(b).unwrap().linear;
    assert!(momentum.length() < 1e-8, "{momentum:?}");
    w.scene
        .set_joint_settings(
            j,
            JointSettings {
                enabled: false,
                contacts_enabled: false,
            },
        )
        .unwrap();
    w.scene
        .set_body_velocity(b, SpatialVector::new(Vec3::X, Vec3::ZERO))
        .unwrap();
    w.step(0.1).unwrap();
    assert!(w.scene.body_pose(b).unwrap().position.x > 0.09);
    w.scene.remove_body(a).unwrap();
    assert!(w.scene.joint(j).is_none());
    assert_eq!(w.scene.loop_joints().count(), 0);
}

#[test]
fn closed_four_bar_moves_without_breaking_closure() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    let ground = body(&mut w, Vec3::ZERO, 1000.0);
    w.scene.set_body_kind(ground, BodyKind::Fixed).unwrap();
    let left = body(&mut w, Vec3::ZERO, 1.0);
    let right = body(&mut w, Vec3::X * 2.0, 1.0);
    let bar = body(&mut w, Vec3::Y, 1.0);
    let mut hinge = desc(Some(ground), left, JointType::Revolute);
    hinge.axis = Vec3::Z;
    let drive = w.scene.insert_joint(hinge.clone()).unwrap();
    hinge.child = right;
    hinge.frame_parent.position = Vec3::X * 2.0;
    w.scene.insert_joint(hinge.clone()).unwrap();
    hinge.parent = Some(left);
    hinge.child = bar;
    hinge.frame_parent.position = Vec3::Y;
    w.scene.insert_joint(hinge.clone()).unwrap();
    hinge.parent = Some(right);
    hinge.frame_child.position = Vec3::X * 2.0;
    assert!(w.scene.insert_joint(hinge.clone()).is_err());
    let closure = w.scene.insert_loop_joint(hinge, softness()).unwrap();
    w.scene
        .set_joint_motor(
            drive,
            JointAxis::AngX,
            JointMotor {
                mode: JointTargetMode::Velocity,
                target_velocity: 0.3,
                damping: 10.0,
                max_force: 10.0,
                ..Default::default()
            },
        )
        .unwrap();
    for _ in 0..120 {
        w.step(1.0 / 120.0).unwrap();
    }
    let j = w.scene.joint(closure).unwrap();
    let p = w
        .scene
        .body_pose(right)
        .unwrap()
        .compose(&j.frame_parent)
        .position;
    let q = w
        .scene
        .body_pose(bar)
        .unwrap()
        .compose(&j.frame_child)
        .position;
    assert!(
        (p - q).length() < 0.005,
        "closure residual {}",
        (p - q).length()
    );
    assert!(
        w.scene
            .body_pose(left)
            .unwrap()
            .rotation
            .angle_between(Quat::IDENTITY)
            > 0.05
    );
    assert!(w.quarantined_bodies().is_empty());
}

#[test]
fn loop_motor_cap_and_world_anchor_are_applied() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    let b = body(&mut w, Vec3::ZERO, 2.0);
    let j = w
        .scene
        .insert_loop_joint(desc(None, b, JointType::Prismatic), softness())
        .unwrap();
    w.scene
        .set_joint_motor(
            j,
            JointAxis::LinX,
            JointMotor {
                mode: JointTargetMode::Velocity,
                target_velocity: 100.0,
                damping: 1000.0,
                max_force: 4.0,
                ..Default::default()
            },
        )
        .unwrap();
    for _ in 0..120 {
        w.step(1.0 / 120.0).unwrap();
    }
    let v = w.scene.body_velocity(b).unwrap();
    assert!((v.linear.x - 2.0).abs() < 1e-8, "{v:?}");
    assert!(v.linear.y.abs() < 1e-9 && v.angular.length() < 1e-9);
}
