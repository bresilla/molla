use molla_math::{Transform, Vec3};
use molla_sim::JointType;
use molla_sim::runtime::{BodyDesc, ColliderDesc, ColliderGeometry, JointDesc, RigidScene};

#[test]
fn structure_stamps_follow_old_and_new_joint_components() {
    let mut scene = RigidScene::new().unwrap();
    let a = scene.insert_body(BodyDesc::default()).unwrap();
    let b = scene.insert_body(BodyDesc::default()).unwrap();
    let prop = scene.insert_body(BodyDesc::default()).unwrap();
    let isolated = scene.body_structure_stamp(prop).unwrap();
    let before_a = scene.body_structure_stamp(a).unwrap();
    let before_b = scene.body_structure_stamp(b).unwrap();
    let joint = scene
        .insert_joint(JointDesc {
            parent: Some(a),
            child: b,
            kind: JointType::Revolute,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
            axis: Vec3::Z,
        })
        .unwrap();
    assert_ne!(scene.body_structure_stamp(a), Some(before_a));
    assert_ne!(scene.body_structure_stamp(b), Some(before_b));
    assert_eq!(scene.body_structure_stamp(prop), Some(isolated.clone()));
    let attached = scene.body_structure_stamp(a).unwrap();
    assert_eq!(scene.body_structure_stamp(b), Some(attached.clone()));
    scene.remove_joint(joint).unwrap();
    assert_ne!(scene.body_structure_stamp(a), Some(attached.clone()));
    assert_ne!(scene.body_structure_stamp(b), Some(attached));
    assert_eq!(scene.body_structure_stamp(prop), Some(isolated));
    let detached = scene.body_structure_stamp(a).unwrap();
    scene.remove_body(b).unwrap();
    assert_eq!(scene.body_structure_stamp(a), Some(detached));
    assert_eq!(scene.body_structure_stamp(b), None);
}

#[test]
fn collider_reparenting_invalidates_both_components_atomically() {
    let mut scene = RigidScene::new().unwrap();
    let a = scene.insert_body(BodyDesc::default()).unwrap();
    let b = scene.insert_body(BodyDesc::default()).unwrap();
    let mut collider = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
    collider.parent = Some(a);
    let shape = scene.insert_collider(collider.clone()).unwrap();
    let before_a = scene.body_structure_stamp(a).unwrap();
    let before_b = scene.body_structure_stamp(b).unwrap();
    collider.parent = Some(b);
    scene.replace_collider(shape, collider.clone()).unwrap();
    let after_a = scene.body_structure_stamp(a).unwrap();
    let after_b = scene.body_structure_stamp(b).unwrap();
    assert_ne!(before_a, after_a);
    assert_ne!(before_b, after_b);
    collider.parent = Some(a);
    collider.geometry = ColliderGeometry::Ball { radius: -1.0 };
    assert!(scene.replace_collider(shape, collider).is_err());
    assert_eq!(scene.body_structure_stamp(a), Some(after_a));
    assert_eq!(scene.body_structure_stamp(b), Some(after_b));
}

#[test]
fn state_edit_identity_distinguishes_edits_from_solver_commits() {
    let mut scene = RigidScene::new().unwrap();
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    let state = scene.state_edit_stamp();
    let structure = scene.body_structure_stamp(body).unwrap();
    let next = scene.state().clone_host().unwrap();
    scene.commit_step(next).unwrap();
    assert_eq!(scene.state_edit_stamp(), state);
    let prop = scene.insert_body(BodyDesc::default()).unwrap();
    scene.remove_body(prop).unwrap();
    assert_eq!(scene.state_edit_stamp(), state);
    assert_eq!(scene.body_structure_stamp(body), Some(structure.clone()));
    scene
        .set_body_pose_with_wake(body, Transform::from_translation(Vec3::X), false)
        .unwrap();
    let teleported = scene.state_edit_stamp();
    assert_ne!(teleported, state);
    assert_eq!(scene.body_structure_stamp(body), Some(structure));
    let _ = scene.state_mut();
    assert_ne!(scene.state_edit_stamp(), teleported);
}
