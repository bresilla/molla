use molla_geometry::mass_properties::set_body_additional_mass_with_wake;
use molla_math::{Mat3, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, ColliderDesc, ColliderGeometry, RigidScene};
use molla_sim::{ColliderMass, MassProperties};

#[test]
fn collider_and_additional_mass_recompute_respects_activation_and_is_atomic() {
    for wake in [false, true] {
        let mut scene = RigidScene::new().unwrap();
        let a = scene.insert_body(BodyDesc::default()).unwrap();
        let b = scene.insert_body(BodyDesc::default()).unwrap();
        let mut collider = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
        collider.parent = Some(a);
        collider.mass = ColliderMass::Mass(2.0);
        collider.pose = Transform::from_translation(Vec3::X);
        scene.insert_collider(collider).unwrap();
        let activation = [scene.activation_stamp(a), scene.activation_stamp(b)];
        let additional = MassProperties {
            mass: 4.0,
            local_com: Vec3::ZERO,
            inertia: Mat3::IDENTITY * 4.0,
        };
        set_body_additional_mass_with_wake(&mut scene, a, additional, wake).unwrap();
        let total = scene.body_mass_properties(a).unwrap();
        assert_eq!(total.mass, 6.0);
        assert!((total.local_com - Vec3::X / 3.0).length() < 1e-10);
        assert_eq!(scene.body(a).unwrap().additional_mass, Some(additional));
        assert_eq!(scene.activation_stamp(a) != activation[0], wake);
        assert_eq!(scene.activation_stamp(b), activation[1]);
        let activation = scene.activation_stamp(a);
        let stamp = scene.stamp();
        set_body_additional_mass_with_wake(&mut scene, a, additional, true).unwrap();
        assert!(
            set_body_additional_mass_with_wake(
                &mut scene,
                a,
                MassProperties {
                    mass: -1.0,
                    ..additional
                },
                true
            )
            .is_err()
        );
        assert_eq!(scene.body_mass_properties(a).unwrap(), total);
        assert_eq!(scene.body(a).unwrap().additional_mass, Some(additional));
        assert_eq!(scene.activation_stamp(a), activation);
        assert_eq!(scene.stamp(), stamp);
    }
}
