use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, JointDesc, RigidScene};
use molla_sim::{JointType, MassProperties};

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
fn generalized_rates_activate_only_changed_joints_and_noop_preserves_stamp() {
    for wake in [false, true] {
        let mut scene = RigidScene::new().unwrap();
        let a = scene.insert_body(dynamic()).unwrap();
        let b = scene.insert_body(dynamic()).unwrap();
        let activation = [scene.activation_stamp(a), scene.activation_stamp(b)];
        let mut rates = scene.state().joint_qd.host().unwrap().to_vec();
        rates[0] = 2.0;
        scene
            .set_generalized_velocities_with_wake(&rates, wake)
            .unwrap();
        assert_eq!(scene.activation_stamp(a) != activation[0], wake);
        assert_eq!(scene.activation_stamp(b), activation[1]);
        assert_eq!(
            scene.body_velocity(a).unwrap(),
            SpatialVector::new(Vec3::X * 2.0, Vec3::ZERO)
        );
        assert_eq!(scene.body_velocity(b).unwrap(), SpatialVector::ZERO);
        let stamp = scene.stamp();
        let revision = scene.revision();
        let activation = scene.activation_stamp(a);
        scene
            .set_generalized_velocities_with_wake(&rates, true)
            .unwrap();
        assert_eq!(scene.stamp(), stamp);
        assert_eq!(scene.revision(), revision);
        assert_eq!(scene.activation_stamp(a), activation);
    }
}

#[test]
fn invalid_rates_are_atomic_including_finite_rates_that_overflow_body_velocity() {
    let mut scene = RigidScene::new().unwrap();
    let a = scene.insert_body(dynamic()).unwrap();
    let mut desc = dynamic();
    desc.params.pose.position = Vec3::X * 3.0;
    let b = scene.insert_body(desc).unwrap();
    scene
        .insert_joint(JointDesc {
            parent: Some(a),
            child: b,
            kind: JointType::Fixed,
            frame_parent: Transform::from_translation(Vec3::X * 3.0),
            frame_child: Transform::IDENTITY,
            axis: Vec3::Z,
        })
        .unwrap();
    let initial = scene.state().joint_qd.host().unwrap().to_vec();
    let stamp = scene.stamp();
    let revision = scene.revision();
    let activation = scene.activation_stamp(a);
    for wake in [false, true] {
        for bad in [f64::NAN, f64::INFINITY, 1e308] {
            let mut rates = initial.clone();
            rates[5] = bad;
            assert!(
                scene
                    .set_generalized_velocities_with_wake(&rates, wake)
                    .is_err()
            );
        }
        assert!(
            scene
                .set_generalized_velocities_with_wake(&[], wake)
                .is_err()
        );
        assert_eq!(scene.state().joint_qd.host().unwrap(), initial);
        assert_eq!(scene.body_velocity(a).unwrap(), SpatialVector::ZERO);
        assert_eq!(scene.body_velocity(b).unwrap(), SpatialVector::ZERO);
        assert_eq!(scene.stamp(), stamp);
        assert_eq!(scene.revision(), revision);
        assert_eq!(scene.activation_stamp(a), activation);
    }
}

#[test]
fn malformed_existing_rate_buffer_is_rejected_without_indexing_it() {
    let mut scene = RigidScene::new().unwrap();
    let a = scene.insert_body(dynamic()).unwrap();
    let valid = scene.state().joint_qd.host().unwrap().to_vec();
    scene.state_mut().joint_qd = molla_sim::State::empty().joint_qd;
    let stamp = scene.activation_stamp(a);
    for wake in [false, true] {
        assert!(
            scene
                .set_generalized_velocities_with_wake(&valid, wake)
                .is_err()
        );
        assert!(scene.state().joint_qd.host().unwrap().is_empty());
        assert_eq!(scene.activation_stamp(a), stamp);
    }
}
