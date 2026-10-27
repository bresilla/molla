use molla_math::{Mat3, SpatialVector, Vec3};
use molla_sim::MassProperties;
use molla_sim::runtime::{BodyDesc, BodyKind, RigidScene};

fn dynamic() -> BodyDesc {
    BodyDesc {
        additional_mass: Some(MassProperties {
            mass: 1.0,
            local_com: Vec3::ZERO,
            inertia: Mat3::IDENTITY,
        }),
        ..Default::default()
    }
}

#[test]
fn velocity_and_wrench_flags_only_change_requested_activation() {
    for wake in [false, true] {
        let mut scene = RigidScene::new().unwrap();
        let a = scene.insert_body(dynamic()).unwrap();
        let b = scene.insert_body(dynamic()).unwrap();
        let other = scene.activation_stamp(b);
        for edit in 0..4 {
            let activation = scene.activation_stamp(a);
            let stamp = scene.stamp();
            match edit {
                0 => scene
                    .set_body_velocity_with_wake(a, SpatialVector::new(Vec3::X, Vec3::Y), wake)
                    .unwrap(),
                1 => scene
                    .add_body_wrench_with_wake(a, SpatialVector::new(Vec3::Y, Vec3::Z), wake)
                    .unwrap(),
                2 => scene
                    .add_body_force_at_point_with_wake(
                        a,
                        Vec3::X,
                        scene.body_world_com(a).unwrap() + Vec3::Y,
                        wake,
                    )
                    .unwrap(),
                _ => scene.reset_body_forces_with_wake(a, wake).unwrap(),
            }
            assert_eq!(scene.activation_stamp(a) != activation, wake, "edit={edit}");
            assert_eq!(scene.activation_stamp(b), other);
            if edit == 0 {
                assert_ne!(scene.stamp(), stamp);
                assert_eq!(
                    scene.body_velocity(a).unwrap(),
                    SpatialVector::new(Vec3::X, Vec3::Y)
                );
            }
            if edit == 2 {
                assert_eq!(
                    scene.state().body_f.host().unwrap()[0],
                    SpatialVector::new(Vec3::X + Vec3::Y, Vec3::ZERO)
                );
            }
        }
        assert_eq!(scene.state().body_f.host().unwrap()[0], SpatialVector::ZERO);
    }
}

#[test]
fn invalid_nonwaking_edits_preserve_state_and_activation() {
    for wake in [false, true] {
        let mut scene = RigidScene::new().unwrap();
        let a = scene.insert_body(dynamic()).unwrap();
        let removed = scene.insert_body(dynamic()).unwrap();
        scene.remove_body(removed).unwrap();
        scene
            .add_body_wrench(a, SpatialVector::new(Vec3::splat(f64::MAX), Vec3::ZERO))
            .unwrap();
        let force = scene.state().body_f.host().unwrap().to_vec();
        let rates = scene.state().joint_qd.host().unwrap().to_vec();
        let activation = scene.activation_stamp(a);
        let stamp = scene.stamp();
        let revision = scene.revision();
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let bad = SpatialVector::new(Vec3::splat(value), Vec3::ZERO);
            assert!(scene.set_body_velocity_with_wake(a, bad, wake).is_err());
            assert!(scene.add_body_wrench_with_wake(a, bad, wake).is_err());
            assert!(
                scene
                    .add_body_force_at_point_with_wake(a, Vec3::X, Vec3::splat(value), wake)
                    .is_err()
            );
        }
        assert!(
            scene
                .add_body_wrench_with_wake(
                    a,
                    SpatialVector::new(Vec3::splat(f64::MAX), Vec3::ZERO),
                    wake
                )
                .is_err()
        );
        assert!(
            scene
                .set_body_velocity_with_wake(removed, SpatialVector::ZERO, wake)
                .is_err()
        );
        assert!(
            scene
                .add_body_wrench_with_wake(removed, SpatialVector::ZERO, wake)
                .is_err()
        );
        assert!(scene.reset_body_forces_with_wake(removed, wake).is_err());
        assert_eq!(scene.state().body_f.host().unwrap(), force);
        assert_eq!(scene.state().joint_qd.host().unwrap(), rates);
        assert_eq!(scene.body_velocity(a).unwrap(), SpatialVector::ZERO);
        assert_eq!(scene.activation_stamp(a), activation);
        assert_eq!(scene.stamp(), stamp);
        assert_eq!(scene.revision(), revision);
    }
}

#[test]
fn fixed_bodies_and_zero_wrenches_do_not_activate() {
    let mut scene = RigidScene::new().unwrap();
    let a = scene.insert_body(dynamic()).unwrap();
    let fixed = scene
        .insert_body(BodyDesc {
            kind: BodyKind::Fixed,
            ..dynamic()
        })
        .unwrap();
    for id in [a, fixed] {
        let stamp = scene.activation_stamp(id);
        for wake in [false, true] {
            scene
                .add_body_wrench_with_wake(id, SpatialVector::ZERO, wake)
                .unwrap();
            scene.reset_body_forces_with_wake(id, wake).unwrap();
            if id == fixed {
                scene
                    .add_body_wrench_with_wake(id, SpatialVector::new(Vec3::X, Vec3::Z), wake)
                    .unwrap();
                scene
                    .set_body_velocity_with_wake(id, SpatialVector::new(Vec3::X, Vec3::Z), wake)
                    .unwrap();
            }
        }
        assert_eq!(scene.activation_stamp(id), stamp);
        assert_eq!(scene.body_velocity(id).unwrap(), SpatialVector::ZERO);
    }
}
