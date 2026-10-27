use molla_math::{Transform, Vec3};
use molla_sim::JointType;
use molla_sim::runtime::{BodyDesc, ColliderDesc, ColliderGeometry, JointDesc, RigidScene};

#[test]
fn insertion_identities_survive_edits_and_forks_but_not_independent_insertions() {
    let mut scene = RigidScene::new().unwrap();
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    let identity = scene.body_identity_stamp(body).unwrap();
    let mut fork = scene.fork().unwrap();
    assert_eq!(fork.body_identity_stamp(body), Some(identity.clone()));
    let a = scene.insert_body(BodyDesc::default()).unwrap();
    let b = fork.insert_body(BodyDesc::default()).unwrap();
    assert_eq!(a, b);
    assert_ne!(scene.body_identity_stamp(a), fork.body_identity_stamp(b));
    scene
        .set_body_pose(body, Transform::from_translation(Vec3::X))
        .unwrap();
    assert_eq!(scene.body_identity_stamp(body), Some(identity.clone()));
    assert!(
        scene
            .set_body_pose(body, Transform::from_translation(Vec3::splat(f64::NAN)))
            .is_err()
    );
    assert_eq!(scene.body_identity_stamp(body), Some(identity));
    scene.remove_body(body).unwrap();
    assert!(scene.body_identity_stamp(body).is_none());
    assert!(fork.body_identity_stamp(body).is_some());
}

#[test]
fn joint_insertion_identity_survives_repair_and_clears_on_cascade_removal() {
    let mut scene = RigidScene::new().unwrap();
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    let joint = scene
        .insert_joint(JointDesc {
            parent: None,
            child: body,
            kind: JointType::Fixed,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    let identity = scene.joint_identity_stamp(joint).unwrap();
    let fork = scene.fork().unwrap();
    assert_eq!(fork.joint_identity_stamp(joint), Some(identity.clone()));
    scene.break_joints(&[joint]).unwrap();
    scene.repair_joint(joint).unwrap();
    scene
        .set_joint_frames(
            joint,
            Transform::from_translation(Vec3::X),
            Transform::IDENTITY,
        )
        .unwrap();
    assert_eq!(scene.joint_identity_stamp(joint), Some(identity));
    scene.remove_body(body).unwrap();
    assert!(scene.joint_identity_stamp(joint).is_none());
    assert!(fork.joint_identity_stamp(joint).is_some());
}

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
    let contact_a = scene.body_contact_stamp(a).unwrap();
    let contact_b = scene.body_contact_stamp(b).unwrap();
    scene.replace_collider(shape, collider.clone()).unwrap();
    let after_a = scene.body_structure_stamp(a).unwrap();
    let after_b = scene.body_structure_stamp(b).unwrap();
    assert_ne!(before_a, after_a);
    assert_ne!(before_b, after_b);
    assert_ne!(scene.body_contact_stamp(a), Some(contact_a));
    assert_ne!(scene.body_contact_stamp(b), Some(contact_b));
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

#[test]
fn body_state_edits_follow_only_the_edited_joint_component() {
    let mut scene = RigidScene::new().unwrap();
    let chassis = scene.insert_body(BodyDesc::default()).unwrap();
    let wheel = scene.insert_body(BodyDesc::default()).unwrap();
    let other = scene.insert_body(BodyDesc::default()).unwrap();
    scene
        .insert_joint(JointDesc {
            parent: Some(chassis),
            child: wheel,
            kind: JointType::Revolute,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
            axis: Vec3::Z,
        })
        .unwrap();
    let original = scene.body_state_stamp(other).unwrap();
    scene
        .set_body_pose_with_wake(chassis, Transform::from_translation(Vec3::X), false)
        .unwrap();
    let edited = scene.body_state_stamp(chassis).unwrap();
    assert_ne!(edited, original);
    assert_eq!(scene.body_state_stamp(wheel), Some(edited.clone()));
    assert_eq!(scene.body_state_stamp(other), Some(original.clone()));
    scene
        .commit_step(scene.state().clone_host().unwrap())
        .unwrap();
    assert_eq!(scene.body_state_stamp(chassis), Some(edited.clone()));
    assert_eq!(scene.body_state_stamp(other), Some(original.clone()));
    assert!(
        scene
            .set_body_pose(wheel, Transform::from_translation(Vec3::splat(f64::NAN)))
            .is_err()
    );
    assert_eq!(scene.body_state_stamp(chassis), Some(edited));
    assert_eq!(scene.body_state_stamp(other), Some(original.clone()));
    let _ = scene.state_mut();
    assert_ne!(scene.body_state_stamp(other), Some(original));
    assert_eq!(
        scene.body_state_stamp(chassis),
        scene.body_state_stamp(other)
    );
    scene.remove_body(wheel).unwrap();
    assert_eq!(scene.body_state_stamp(wheel), None);
}

#[test]
fn contact_stamps_distinguish_loop_forces_from_contact_response_edits() {
    use molla_sim::joint_control::{JointAxes, JointAxis, JointSoftness};
    use molla_sim::runtime::JointSettings;

    let mut scene = RigidScene::new().unwrap();
    let a = scene.insert_body(BodyDesc::default()).unwrap();
    let b = scene.insert_body(BodyDesc::default()).unwrap();
    let prop = scene.insert_body(BodyDesc::default()).unwrap();
    let joint = scene
        .insert_loop_joint(
            JointDesc {
                parent: Some(a),
                child: b,
                kind: JointType::D6,
                axis: Vec3::X,
                frame_parent: Transform::IDENTITY,
                frame_child: Transform::IDENTITY,
            },
            JointSoftness {
                natural_frequency: 4.0,
                damping_ratio: 0.6,
            },
        )
        .unwrap();
    let original = [a, b, prop].map(|body| scene.body_contact_stamp(body).unwrap());
    let structure = scene.body_structure_stamp(b).unwrap();
    scene
        .set_joint_settings(
            joint,
            JointSettings {
                enabled: false,
                contacts_enabled: false,
            },
        )
        .unwrap();
    assert_ne!(scene.body_structure_stamp(b), Some(structure));
    assert_eq!(
        [a, b, prop].map(|body| scene.body_contact_stamp(body).unwrap()),
        original
    );
    scene
        .set_joint_settings(joint, JointSettings::default())
        .unwrap();
    scene
        .set_joint_frames(
            joint,
            Transform::from_translation(Vec3::X),
            Transform::IDENTITY,
        )
        .unwrap();
    scene
        .set_joint_locked_axes(joint, JointAxes::NONE.with(JointAxis::LinX))
        .unwrap();
    assert_eq!(
        [a, b, prop].map(|body| scene.body_contact_stamp(body).unwrap()),
        original
    );
    let mut fork = scene.fork().unwrap();
    assert_eq!(fork.body_contact_stamp(b), Some(original[1].clone()));
    assert!(
        scene
            .set_joint_frames(
                joint,
                Transform::from_translation(Vec3::splat(f64::NAN)),
                Transform::IDENTITY,
            )
            .is_err()
    );
    assert_eq!(scene.body_contact_stamp(b), Some(original[1].clone()));
    scene
        .set_joint_settings(
            joint,
            JointSettings {
                contacts_enabled: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_ne!(scene.body_contact_stamp(a), Some(original[0].clone()));
    assert_ne!(scene.body_contact_stamp(b), Some(original[1].clone()));
    assert_eq!(scene.body_contact_stamp(prop), Some(original[2].clone()));
    let before = fork.body_contact_stamp(b).unwrap();
    fork.set_body_mass_properties(
        b,
        molla_sim::MassProperties {
            mass: 2.0,
            local_com: Vec3::ZERO,
            inertia: molla_math::Mat3::IDENTITY * 2.0,
        },
    )
    .unwrap();
    assert_ne!(fork.body_contact_stamp(a), Some(original[0].clone()));
    assert_ne!(fork.body_contact_stamp(b), Some(before));
    assert_eq!(fork.body_contact_stamp(prop), Some(original[2].clone()));
    let before = fork.body_contact_stamp(b).unwrap();
    fork.remove_joint(joint).unwrap();
    assert_ne!(fork.body_contact_stamp(b), Some(before));
    fork.remove_body(b).unwrap();
    assert!(fork.body_contact_stamp(b).is_none());
}

#[test]
fn rigid_compliance_transitions_invalidate_contact_response_identity() {
    use molla_sim::joint_control::JointSoftness;

    let mut scene = RigidScene::new().unwrap();
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    let joint = scene
        .insert_loop_joint(
            JointDesc {
                parent: None,
                child: body,
                kind: JointType::Ball,
                axis: Vec3::X,
                frame_parent: Transform::IDENTITY,
                frame_child: Transform::IDENTITY,
            },
            JointSoftness {
                natural_frequency: 4.0,
                damping_ratio: 0.6,
            },
        )
        .unwrap();
    let soft = scene.body_contact_stamp(body).unwrap();
    scene.set_joint_softness(joint, None).unwrap();
    let rigid = scene.body_contact_stamp(body).unwrap();
    assert_ne!(rigid, soft);
    scene
        .set_joint_softness(
            joint,
            Some(JointSoftness {
                natural_frequency: 7.0,
                damping_ratio: 0.6,
            }),
        )
        .unwrap();
    assert_ne!(scene.body_contact_stamp(body), Some(rigid));
    let soft = scene.body_contact_stamp(body).unwrap();
    scene
        .set_cone_limits(
            joint,
            molla_sim::joint_control::SphericalConeLimits {
                angle0: Some(0.5),
                angle1: Some(0.4),
            },
        )
        .unwrap();
    assert_ne!(scene.body_contact_stamp(body), Some(soft));
    let cone = scene.body_contact_stamp(body).unwrap();
    scene
        .set_joint_frames(
            joint,
            Transform::new(Vec3::ZERO, molla_math::Quat::from_rotation_y(0.1)),
            Transform::IDENTITY,
        )
        .unwrap();
    assert_ne!(scene.body_contact_stamp(body), Some(cone));
}

#[test]
fn joint_response_edits_retain_episode_stamps_while_physical_configuration_invalidates_them() {
    let mut scene = RigidScene::new().unwrap();
    let a = scene.insert_body(BodyDesc::default()).unwrap();
    let b = scene.insert_body(BodyDesc::default()).unwrap();
    let prop = scene.insert_body(BodyDesc::default()).unwrap();
    let root = scene
        .insert_joint(JointDesc {
            parent: None,
            child: a,
            kind: JointType::Prismatic,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    scene
        .insert_joint(JointDesc {
            parent: Some(a),
            child: b,
            kind: JointType::Fixed,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    let mut collider = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.2 });
    collider.parent = Some(b);
    let shape = scene.insert_collider(collider).unwrap();
    let original = [a, b, prop].map(|body| scene.body_contact_episode_stamp(body).unwrap());
    let response = scene.body_contact_stamp(b).unwrap();
    scene.break_joints(&[root]).unwrap();
    assert_ne!(scene.body_contact_stamp(b), Some(response));
    assert_eq!(
        [a, b, prop].map(|body| scene.body_contact_episode_stamp(body).unwrap()),
        original
    );
    let response = scene.body_contact_stamp(b).unwrap();
    scene.repair_joint(root).unwrap();
    assert_ne!(scene.body_contact_stamp(b), Some(response));
    assert_eq!(
        [a, b, prop].map(|body| scene.body_contact_episode_stamp(body).unwrap()),
        original
    );
    scene
        .set_joint_frames(
            root,
            Transform::from_translation(Vec3::Y),
            Transform::IDENTITY,
        )
        .unwrap();
    assert_eq!(
        [a, b, prop].map(|body| scene.body_contact_episode_stamp(body).unwrap()),
        original
    );
    let fork = scene.fork().unwrap();
    assert_eq!(
        fork.body_contact_episode_stamp(b),
        Some(original[1].clone())
    );
    scene
        .set_body_mass_properties(
            b,
            molla_sim::MassProperties {
                mass: 2.0,
                local_com: Vec3::ZERO,
                inertia: molla_math::Mat3::IDENTITY,
            },
        )
        .unwrap();
    assert_ne!(
        scene.body_contact_episode_stamp(a),
        Some(original[0].clone())
    );
    assert_ne!(
        scene.body_contact_episode_stamp(b),
        Some(original[1].clone())
    );
    assert_eq!(
        scene.body_contact_episode_stamp(prop),
        Some(original[2].clone())
    );
    let mass = scene.body_contact_episode_stamp(b).unwrap();
    let mut material = scene.collider(shape).unwrap().clone();
    material.restitution = 0.25;
    scene.replace_collider(shape, material).unwrap();
    assert_ne!(scene.body_contact_episode_stamp(b), Some(mass));
    let material = scene.body_contact_episode_stamp(b).unwrap();
    let state = scene.body_state_stamp(b).unwrap();
    scene
        .set_body_pose(b, Transform::from_translation(Vec3::Z))
        .unwrap();
    assert_eq!(scene.body_contact_episode_stamp(b), Some(material));
    assert_ne!(scene.body_state_stamp(b), Some(state));
    let before = scene.body_contact_episode_stamp(b).unwrap();
    let mut invalid = scene.collider(shape).unwrap().clone();
    invalid.geometry = ColliderGeometry::Ball { radius: -1.0 };
    assert!(scene.replace_collider(shape, invalid).is_err());
    assert_eq!(scene.body_contact_episode_stamp(b), Some(before));
    scene.remove_body(b).unwrap();
    assert!(scene.body_contact_episode_stamp(b).is_none());
    assert_eq!(
        fork.body_contact_episode_stamp(b),
        Some(original[1].clone())
    );
}

#[test]
fn a_revolute_joint_at_rest_reads_zero_whatever_its_frame_quaternion_sign() {
    let mut scene = RigidScene::new().unwrap();
    let a = scene.insert_body(BodyDesc::default()).unwrap();
    let b = scene.insert_body(BodyDesc::default()).unwrap();
    let joint = scene
        .insert_joint(JointDesc {
            parent: Some(a),
            child: b,
            kind: JointType::Revolute,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::new(Vec3::ZERO, -molla_math::Quat::IDENTITY),
            axis: Vec3::Z,
        })
        .unwrap();
    let index = scene.joint_index(joint).unwrap().index();
    let coordinate = scene.model().joint_coord_offset.host().unwrap()[index] as usize;
    assert_eq!(scene.state().joint_q.host().unwrap()[coordinate], 0.0);
}
