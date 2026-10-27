use molla_math::{Mat3, SpatialVector, Vec3};
use molla_sim::MassProperties;
use molla_sim::runtime::{BodyDesc, RigidScene};

fn mass(value: f64) -> MassProperties {
    MassProperties {
        mass: value,
        local_com: Vec3::ZERO,
        inertia: Mat3::IDENTITY * value,
    }
}

#[test]
fn mass_rebuilds_activate_only_the_requested_body_and_noops_preserve_identity() {
    for wake in [false, true] {
        for components in [false, true] {
            let mut scene = RigidScene::new().unwrap();
            let a = scene.insert_body(BodyDesc::default()).unwrap();
            let b = scene.insert_body(BodyDesc::default()).unwrap();
            let activation = [scene.activation_stamp(a), scene.activation_stamp(b)];
            let before = scene.stamp();
            let replacement = mass(3.0);
            if components {
                scene
                    .set_body_mass_components_with_wake(a, replacement, replacement, wake)
                    .unwrap();
            } else {
                scene
                    .set_body_mass_properties_with_wake(a, replacement, wake)
                    .unwrap();
            }
            assert_ne!(scene.stamp(), before);
            assert_eq!(scene.body_mass_properties(a).unwrap(), replacement);
            assert_eq!(scene.activation_stamp(a) != activation[0], wake);
            assert_eq!(scene.activation_stamp(b), activation[1]);
            let stamp = scene.stamp();
            let revision = scene.revision();
            let activation = scene.activation_stamp(a);
            if components {
                scene
                    .set_body_mass_components_with_wake(a, replacement, replacement, true)
                    .unwrap();
            } else {
                scene
                    .set_body_mass_properties_with_wake(a, replacement, true)
                    .unwrap();
            }
            assert_eq!(scene.stamp(), stamp);
            assert_eq!(scene.revision(), revision);
            assert_eq!(scene.activation_stamp(a), activation);
        }
    }
}

#[test]
fn nonwaking_com_edit_preserves_velocity_field_and_wrench_application_point() {
    let mut scene = RigidScene::new().unwrap();
    let a = scene
        .insert_body(BodyDesc {
            velocity: SpatialVector::new(Vec3::X, Vec3::Z * 2.0),
            additional_mass: Some(mass(2.0)),
            ..Default::default()
        })
        .unwrap();
    let b = scene.insert_body(BodyDesc::default()).unwrap();
    let force = SpatialVector::new(Vec3::Y * 3.0, Vec3::X);
    scene.add_body_wrench(a, force).unwrap();
    scene.control_mut().body_external_f.host_mut().unwrap()[0] = force;
    let activation = [scene.activation_stamp(a), scene.activation_stamp(b)];
    let mut rates = scene.state().joint_qd.host().unwrap().to_vec();
    let point = Vec3::new(1.0, 2.0, 3.0);
    let velocity = scene.body_velocity_at_point(a, point).unwrap();
    let replacement = MassProperties {
        local_com: Vec3::X * 0.5,
        ..mass(4.0)
    };
    scene
        .set_body_mass_components_with_wake(a, replacement, replacement, false)
        .unwrap();
    assert_eq!(scene.body_velocity_at_point(a, point).unwrap(), velocity);
    rates[..3].copy_from_slice(&scene.body_velocity(a).unwrap().linear.to_array());
    assert_eq!(scene.state().joint_qd.host().unwrap(), rates);
    for actual in [
        scene.state().body_f.host().unwrap()[0],
        scene.control().body_external_f.host().unwrap()[0],
    ] {
        assert_eq!(actual.linear, force.linear);
        assert_eq!(
            actual.angular + replacement.local_com.cross(force.linear),
            force.angular
        );
    }
    assert_eq!(
        [scene.activation_stamp(a), scene.activation_stamp(b)],
        activation
    );
}

#[test]
fn rejected_mass_edits_leave_descriptors_state_and_activation_unchanged() {
    for wake in [false, true] {
        let mut scene = RigidScene::new().unwrap();
        let a = scene
            .insert_body(BodyDesc {
                velocity: SpatialVector::new(Vec3::ZERO, Vec3::Z * 2.0),
                additional_mass: Some(mass(2.0)),
                ..Default::default()
            })
            .unwrap();
        let removed = scene.insert_body(BodyDesc::default()).unwrap();
        scene.remove_body(removed).unwrap();
        let stamp = scene.stamp();
        let activation = scene.activation_stamp(a);
        let velocity = scene.body_velocity(a).unwrap();
        for bad in [
            mass(-1.0),
            mass(f64::NAN),
            MassProperties {
                local_com: Vec3::X * 1e308,
                ..mass(3.0)
            },
        ] {
            assert!(
                scene
                    .set_body_mass_properties_with_wake(a, bad, wake)
                    .is_err()
            );
            assert!(
                scene
                    .set_body_mass_components_with_wake(a, bad, bad, wake)
                    .is_err()
            );
        }
        assert!(
            scene
                .set_body_mass_components_with_wake(a, mass(3.0), mass(-1.0), wake)
                .is_err()
        );
        assert!(
            scene
                .set_body_mass_properties_with_wake(removed, mass(3.0), wake)
                .is_err()
        );
        assert_eq!(scene.body_mass_properties(a).unwrap(), mass(2.0));
        assert_eq!(scene.body(a).unwrap().additional_mass, Some(mass(2.0)));
        assert_eq!(scene.body_velocity(a).unwrap(), velocity);
        assert_eq!(scene.stamp(), stamp);
        assert_eq!(scene.activation_stamp(a), activation);
    }
}
