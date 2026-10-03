use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxis, JointMotor, MotorModel};
use molla_sim::runtime::{
    BodyDesc, ColliderDesc, ColliderGeometry, DistanceLimits, JointDesc, RigidScene,
    RigidSceneBuilder, SphericalConeLimits,
};
use molla_sim::{BodyParams, ColliderMass, JointTargetMode, JointType, MassProperties};

macro_rules! populate {
    ($scene:ident, $kind:expr) => {{
        $scene.set_gravity(Vec3::new(1., -10., 2.)).unwrap();
        let body = $scene
            .insert_body(BodyDesc {
                params: BodyParams::new().with_position(Vec3::new(1., 2., 3.)),
                velocity: SpatialVector::new(Vec3::X * 0.1, Vec3::Z * 0.2),
                additional_mass: Some(MassProperties {
                    mass: 17.,
                    local_com: Vec3::new(0.1, 0.2, 0.3),
                    inertia: Mat3::from_cols(
                        Vec3::new(2., 0.1, 0.2),
                        Vec3::new(0.1, 3., 0.3),
                        Vec3::new(0.2, 0.3, 4.),
                    ),
                }),
                ..Default::default()
            })
            .unwrap();
        let mut collider = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
        collider.parent = Some(body);
        collider.mass = ColliderMass::Mass(0.);
        let shape = $scene.insert_collider(collider).unwrap();
        let joint = $scene
            .insert_joint(JointDesc {
                parent: None,
                child: body,
                kind: $kind,
                axis: Vec3::X,
                frame_parent: Transform::from_translation(Vec3::new(1., 2., 3.)),
                frame_child: Transform::new(Vec3::ZERO, Quat::from_rotation_z(0.3)),
            })
            .unwrap();
        match $kind {
            JointType::Distance => $scene
                .set_distance_limits(
                    joint,
                    DistanceLimits {
                        min: Some(0.5),
                        max: Some(1.5),
                    },
                )
                .unwrap(),
            JointType::Ball => $scene
                .set_cone_limits(
                    joint,
                    SphericalConeLimits {
                        angle0: Some(0.25),
                        angle1: Some(0.4),
                    },
                )
                .unwrap(),
            JointType::D6 => {
                for axis in [
                    JointAxis::LinY,
                    JointAxis::LinZ,
                    JointAxis::AngX,
                    JointAxis::AngY,
                    JointAxis::AngZ,
                ] {
                    $scene
                        .set_joint_limits(joint, axis, Some([0., 0.]))
                        .unwrap();
                }
                $scene
                    .set_joint_motor(
                        joint,
                        JointAxis::LinX,
                        JointMotor {
                            mode: JointTargetMode::Velocity,
                            model: MotorModel::Acceleration,
                            target_velocity: 0.1,
                            damping: 10.,
                            max_force: 20.,
                            ..Default::default()
                        },
                    )
                    .unwrap();
            }
            _ => {}
        }
        (body, shape, joint)
    }};
}

#[test]
fn staged_and_sequential_authoring_compile_identical_dynamics() {
    for kind in [
        JointType::Distance,
        JointType::Ball,
        JointType::D6,
        JointType::Fixed,
    ] {
        let mut sequential = RigidScene::new().unwrap();
        let expected = populate!(sequential, kind);
        let mut builder = RigidSceneBuilder::new();
        let handles = populate!(builder, kind);
        assert_eq!(handles, expected);
        let mut staged = builder.build().unwrap();
        macro_rules! equal {
            ($part:ident, $($field:ident),+) => {$(
                assert_eq!(staged.$part().$field.host().unwrap(), sequential.$part().$field.host().unwrap(), stringify!($field));
            )+};
        }
        equal!(
            model,
            gravity,
            body_mass,
            body_com,
            body_inertia,
            body_inv_inertia,
            joint_type,
            joint_x_p,
            joint_x_c,
            joint_dof_offset,
            joint_limit_lower,
            joint_limit_upper,
            shape_geom_kind,
            shape_body,
            shape_transform,
            shape_flags
        );
        equal!(state, body_q, body_qd, body_f, joint_q, joint_qd);
        equal!(
            control,
            joint_target_mode,
            joint_target_pos,
            joint_target_vel,
            joint_target_kp,
            joint_target_kd,
            joint_motor_model,
            joint_motor_max_force,
            joint_force
        );
        assert_eq!(staged.revision(), 0);
        assert!(staged.body_identity_stamp(handles.0).is_some());
        assert!(staged.body_structure_stamp(handles.0).is_some());
        assert!(staged.body_contact_stamp(handles.0).is_some());
        assert!(staged.body_contact_episode_stamp(handles.0).is_some());
        assert!(staged.joint_identity_stamp(handles.2).is_some());
        assert_eq!(
            staged.collider_index(handles.1),
            sequential.collider_index(expected.1)
        );
        let identity = staged.body_identity_stamp(handles.0).unwrap();
        staged.remove_joint(handles.2).unwrap();
        assert_eq!(staged.body_identity_stamp(handles.0).unwrap(), identity);
        let next = staged.insert_body(BodyDesc::default()).unwrap();
        assert!(next.to_bits() > handles.2.to_bits());
        assert_eq!(staged.revision(), 2);
    }
}

#[test]
fn final_build_rejects_invalid_geometry_and_topology() {
    let mut builder = RigidSceneBuilder::new();
    builder
        .insert_collider(ColliderDesc::new(ColliderGeometry::Ball { radius: -1. }))
        .unwrap();
    assert!(builder.build().is_err());
    let mut builder = RigidSceneBuilder::new();
    let body = builder.insert_body(BodyDesc::default()).unwrap();
    builder
        .insert_joint(JointDesc {
            parent: Some(body),
            child: body,
            kind: JointType::Fixed,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    assert!(builder.build().is_err());
}

#[test]
fn failed_settings_do_not_overwrite_valid_staged_inputs() {
    let mut builder = RigidSceneBuilder::new();
    let body = builder.insert_body(BodyDesc::default()).unwrap();
    let joint = builder
        .insert_joint(JointDesc {
            parent: None,
            child: body,
            kind: JointType::D6,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    builder
        .set_joint_limits(joint, JointAxis::LinX, Some([-1., 2.]))
        .unwrap();
    assert!(
        builder
            .set_joint_limits(joint, JointAxis::LinX, Some([2., -1.]))
            .is_err()
    );
    assert!(
        builder
            .set_distance_limits(joint, DistanceLimits::default())
            .is_err()
    );
    assert!(
        builder
            .set_cone_limits(joint, SphericalConeLimits::default())
            .is_err()
    );
    assert!(
        builder
            .set_joint_motor(
                joint,
                JointAxis::LinX,
                JointMotor {
                    damping: -1.,
                    ..Default::default()
                }
            )
            .is_err()
    );
    assert!(builder.set_gravity(Vec3::splat(f64::NAN)).is_err());
    let scene = builder.build().unwrap();
    assert_eq!(
        scene
            .joint_axis_settings(joint, JointAxis::LinX)
            .unwrap()
            .limits,
        Some([-1., 2.])
    );
}

#[test]
fn reverse_body_indices_follow_compaction_forks_and_rejected_edits() {
    let mut builder = RigidSceneBuilder::new();
    let bodies = [0, 1, 2].map(|_| builder.insert_body(BodyDesc::default()).unwrap());
    let mut scene = builder.build().unwrap();
    let verify = |scene: &RigidScene, expected: &[molla_sim::runtime::BodyHandle]| {
        for (index, &body) in expected.iter().enumerate() {
            let dense = molla_core::BodyId::from_index(index);
            assert_eq!(scene.body_handle(dense), Some(body));
            assert_eq!(scene.body_index(body), Some(dense));
        }
        assert_eq!(
            scene.body_handle(molla_core::BodyId::from_index(expected.len())),
            None
        );
    };
    verify(&scene, &bodies);
    scene.remove_body(bodies[1]).unwrap();
    verify(&scene, &[bodies[0], bodies[2]]);
    verify(&scene.fork().unwrap(), &[bodies[0], bodies[2]]);
    let stamp = scene.model_stamp();
    let mut invalid = BodyDesc::default();
    invalid.params.pose.rotation = Quat::from_xyzw(0., 0., 0., 0.);
    assert!(scene.insert_body(invalid).is_err());
    assert_eq!(scene.model_stamp(), stamp);
    verify(&scene, &[bodies[0], bodies[2]]);
    let added = scene.insert_body(BodyDesc::default()).unwrap();
    assert!(!bodies.contains(&added));
    verify(&scene, &[bodies[0], bodies[2], added]);
}
