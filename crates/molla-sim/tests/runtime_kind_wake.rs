use molla_math::{SpatialVector, Vec3};
use molla_sim::runtime::{BodyDesc, BodyKind, RigidScene};

#[test]
fn same_body_kind_is_a_true_noop() {
    let mut scene = RigidScene::new().unwrap();
    let a = scene.insert_body(BodyDesc::default()).unwrap();
    let stamp = scene.stamp();
    let activation = scene.activation_stamp(a);
    scene.set_body_kind(a, BodyKind::Dynamic).unwrap();
    assert_eq!(scene.stamp(), stamp);
    assert_eq!(scene.activation_stamp(a), activation);
}

#[test]
fn body_kind_edit_does_not_activate_unrelated_bodies() {
    let mut scene = RigidScene::new().unwrap();
    let a = scene.insert_body(BodyDesc::default()).unwrap();
    let b = scene.insert_body(BodyDesc::default()).unwrap();
    let activation = scene.activation_stamp(b);
    scene.set_body_kind(a, BodyKind::Kinematic).unwrap();
    assert_eq!(scene.activation_stamp(b), activation);
}

#[test]
fn kind_transition_activation_and_fixed_velocity_follow_wake_policy() {
    let kinds = [BodyKind::Dynamic, BodyKind::Fixed, BodyKind::Kinematic];
    for from in kinds {
        for to in kinds {
            for wake in [false, true] {
                let mut scene = RigidScene::new().unwrap();
                let a = scene
                    .insert_body(BodyDesc {
                        kind: from,
                        velocity: SpatialVector::new(Vec3::X, Vec3::Z),
                        ..Default::default()
                    })
                    .unwrap();
                let b = scene.insert_body(BodyDesc::default()).unwrap();
                let activation = [scene.activation_stamp(a), scene.activation_stamp(b)];
                let pose = scene.body_pose(a).unwrap();
                let velocity = scene.body_velocity(a).unwrap();
                let mass = scene.body_mass_properties(a).unwrap();
                scene.set_body_kind_with_wake(a, to, wake).unwrap();
                assert_eq!(scene.body(a).unwrap().kind, to);
                assert_eq!(scene.body_pose(a).unwrap(), pose);
                assert_eq!(scene.body_mass_properties(a).unwrap(), mass);
                assert_eq!(
                    scene.body_velocity(a).unwrap(),
                    if to == BodyKind::Fixed {
                        SpatialVector::ZERO
                    } else {
                        velocity
                    }
                );
                assert_eq!(
                    scene.activation_stamp(a) != activation[0],
                    from != to && to != BodyKind::Fixed && wake
                );
                assert_eq!(scene.activation_stamp(b), activation[1]);
                let stamp = scene.stamp();
                let revision = scene.revision();
                scene.set_body_kind_with_wake(a, to, true).unwrap();
                assert_eq!(scene.stamp(), stamp);
                assert_eq!(scene.revision(), revision);
            }
        }
    }
}

#[test]
fn unknown_body_kind_edit_is_atomic() {
    let mut scene = RigidScene::new().unwrap();
    let a = scene.insert_body(BodyDesc::default()).unwrap();
    let removed = scene.insert_body(BodyDesc::default()).unwrap();
    scene.remove_body(removed).unwrap();
    let stamp = scene.stamp();
    let activation = scene.activation_stamp(a);
    for wake in [false, true] {
        assert!(
            scene
                .set_body_kind_with_wake(removed, BodyKind::Fixed, wake)
                .is_err()
        );
        assert_eq!(scene.stamp(), stamp);
        assert_eq!(scene.activation_stamp(a), activation);
    }
}
