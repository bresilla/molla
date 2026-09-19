use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{
    BodyDesc, BodyKind, ColliderDesc, ColliderGeometry, JointDesc, RigidScene,
};
use molla_sim::{BodyFlags, JointType, eval_body_velocities, eval_fk};

fn collider(parent: molla_sim::runtime::BodyHandle) -> ColliderDesc {
    ColliderDesc {
        parent: Some(parent),
        geometry: ColliderGeometry::Ball { radius: 0.5 },
        pose: Transform::IDENTITY,
        friction: 0.7,
        restitution: 0.1,
        user_tag: 91,
        ..ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 })
    }
}

#[test]
fn removal_compacts_indices_without_reusing_handles_or_losing_state() {
    let mut scene = RigidScene::new().unwrap();
    let first = scene.insert_body(BodyDesc::default()).unwrap();
    let second = scene.insert_body(BodyDesc::default()).unwrap();
    let shape = scene.insert_collider(collider(second)).unwrap();
    scene.state_mut().joint_q.host_mut().unwrap()[7] = 12.0;
    scene.state_mut().joint_qd.host_mut().unwrap()[6] = 3.0;
    scene.state_mut().body_q.host_mut().unwrap()[1].position.x = 12.0;
    scene.state_mut().body_qd.host_mut().unwrap()[1].linear.x = 3.0;
    scene.state_mut().body_f.host_mut().unwrap()[1].angular.z = 4.0;
    scene.control_mut().joint_force.host_mut().unwrap()[6] = 17.0;
    scene.control_mut().body_external_f.host_mut().unwrap()[1]
        .linear
        .y = 8.0;
    scene.remove_body(first).unwrap();
    assert!(scene.body_index(first).is_none());
    assert_eq!(scene.body_index(second).unwrap().index(), 0);
    assert_eq!(scene.collider_index(shape).unwrap().index(), 0);
    assert_eq!(scene.state().body_q.host().unwrap()[0].position.x, 12.0);
    assert_eq!(scene.state().body_qd.host().unwrap()[0].linear.x, 3.0);
    assert_eq!(scene.state().body_f.host().unwrap()[0].angular.z, 4.0);
    assert_eq!(scene.state().joint_q.host().unwrap()[0], 12.0);
    assert_eq!(scene.state().joint_qd.host().unwrap()[0], 3.0);
    assert_eq!(scene.control().joint_force.host().unwrap()[0], 17.0);
    assert_eq!(
        scene.control().body_external_f.host().unwrap()[0].linear.y,
        8.0
    );
    let next = scene.insert_body(BodyDesc::default()).unwrap();
    assert_ne!(next, first);
    assert_ne!(next, second);
    assert!(scene.remove_body(first).is_err());
    assert_eq!(scene.model().body_count, 2);
}

#[test]
fn rejected_edits_are_atomic() {
    let mut scene = RigidScene::new().unwrap();
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    let revision = scene.revision();
    let mut desc = collider(body);
    desc.geometry = ColliderGeometry::Ball { radius: -1.0 };
    assert!(scene.insert_collider(desc).is_err());
    assert_eq!(scene.model().shape_count, 0);
    assert_eq!(scene.revision(), revision);
    let mut desc = BodyDesc::default();
    desc.params.pose.rotation = Quat::from_xyzw(0.0, 0.0, 0.0, 0.0);
    assert!(scene.insert_body(desc).is_err());
    assert_eq!(scene.model().body_count, 1);
    assert_eq!(scene.revision(), revision);
}

#[test]
fn fixed_mass_is_retained_but_inverse_mass_is_zero() {
    let mut scene = RigidScene::new().unwrap();
    let mut desc = BodyDesc::default();
    desc.params.mass = 3800.0;
    desc.params.inertia = Mat3::from_diagonal(Vec3::splat(200.0));
    desc.velocity = SpatialVector::new(Vec3::X, Vec3::Z);
    let body = scene.insert_body(desc).unwrap();
    scene.set_body_kind(body, BodyKind::Fixed).unwrap();
    assert_eq!(
        scene.model().body_flags.host().unwrap()[0],
        BodyFlags::FIXED
    );
    assert_eq!(scene.model().body_mass.host().unwrap()[0], 3800.0);
    assert_eq!(scene.model().body_inv_mass.host().unwrap()[0], 0.0);
    assert_eq!(
        scene.model().body_inv_inertia.host().unwrap()[0],
        Mat3::ZERO
    );
    assert_eq!(
        scene.state().body_qd.host().unwrap()[0],
        SpatialVector::ZERO
    );
    assert_eq!(scene.model().joint_dof_total, 0);
    scene.set_body_kind(body, BodyKind::Dynamic).unwrap();
    assert_eq!(scene.model().body_inv_mass.host().unwrap()[0], 1.0 / 3800.0);
    assert_eq!(scene.model().joint_dof_total, 6);
}

#[test]
fn merging_and_splitting_preserves_unaffected_joint_state_and_controls() {
    let mut scene = RigidScene::new().unwrap();
    let a = scene.insert_body(BodyDesc::default()).unwrap();
    let b = scene.insert_body(BodyDesc::default()).unwrap();
    let c = scene.insert_body(BodyDesc::default()).unwrap();
    let joint_desc = |parent, child| JointDesc {
        parent: Some(parent),
        child,
        kind: JointType::Revolute,
        frame_parent: Transform::IDENTITY,
        frame_child: Transform::IDENTITY,
        axis: Vec3::Y,
    };
    let bc = scene.insert_joint(joint_desc(b, c)).unwrap();
    let index = scene.joint_index(bc).unwrap().index();
    let q = scene.model().joint_coord_offset.host().unwrap()[index] as usize;
    let v = scene.model().joint_dof_offset.host().unwrap()[index] as usize;
    scene.state_mut().joint_q.host_mut().unwrap()[q] = 0.4;
    scene.state_mut().joint_qd.host_mut().unwrap()[v] = 0.8;
    scene.control_mut().joint_target_kp.host_mut().unwrap()[v] = 70.0;
    let ab = scene.insert_joint(joint_desc(a, b)).unwrap();
    assert_eq!(scene.model().articulation_count, 1);
    let i = scene.joint_index(bc).unwrap().index();
    let q = scene.model().joint_coord_offset.host().unwrap()[i] as usize;
    let v = scene.model().joint_dof_offset.host().unwrap()[i] as usize;
    assert_eq!(scene.state().joint_q.host().unwrap()[q], 0.4);
    assert_eq!(scene.state().joint_qd.host().unwrap()[v], 0.8);
    assert_eq!(scene.control().joint_target_kp.host().unwrap()[v], 70.0);
    let revision = scene.revision();
    assert!(scene.insert_joint(joint_desc(c, a)).is_err());
    assert_eq!(scene.revision(), revision);
    scene.remove_joint(ab).unwrap();
    assert_eq!(scene.model().articulation_count, 2);
    assert!(scene.joint_index(bc).is_some());
    assert!(scene.joint_index(ab).is_none());
    scene.remove_body(b).unwrap();
    assert!(scene.joint_index(bc).is_none());
    assert_eq!(scene.model().articulation_count, 2);
}

#[test]
fn collider_edits_and_cascading_removal_preserve_body_state() {
    let mut scene = RigidScene::new().unwrap();
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    let shape = scene.insert_collider(collider(body)).unwrap();
    let mut desc = collider(body);
    desc.geometry = ColliderGeometry::Cylinder {
        half_height: 0.5,
        radius: 0.7,
    };
    desc.friction = 0.3;
    scene.replace_collider(shape, desc).unwrap();
    assert_eq!(scene.model().shape_friction.host().unwrap()[0], 0.3);
    scene.remove_collider(shape).unwrap();
    assert!(scene.collider_index(shape).is_none());
    let other = scene.insert_collider(collider(body)).unwrap();
    assert_ne!(other, shape);
    scene.remove_body(body).unwrap();
    assert!(scene.collider_index(other).is_none());
    assert_eq!(scene.model().shape_count, 0);
}

#[test]
fn free_root_coordinates_preserve_offset_com_twist() {
    let mut scene = RigidScene::new().unwrap();
    let mut desc = BodyDesc::default();
    desc.params.pose = Transform::new(Vec3::new(12.0, 3.0, -4.0), Quat::from_rotation_y(0.5));
    desc.params.com = Vec3::new(0.5, 0.2, 0.1);
    desc.velocity = SpatialVector::new(Vec3::new(2.0, 0.3, -1.0), Vec3::new(0.2, 1.0, 0.5));
    let pose = desc.params.pose;
    let velocity = desc.velocity;
    scene.insert_body(desc).unwrap();
    let mut state = scene.state().clone_host().unwrap();
    eval_fk(scene.model(), &mut state).unwrap();
    eval_body_velocities(scene.model(), &mut state).unwrap();
    assert!((state.body_q.host().unwrap()[0].position - pose.position).length() < 1e-12);
    let actual = state.body_qd.host().unwrap()[0];
    assert!((actual.linear - velocity.linear).length() < 1e-12);
    assert!((actual.angular - velocity.angular).length() < 1e-12);
}
