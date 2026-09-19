use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, BodyKind, JointDesc, RigidScene};
use molla_sim::{JointType, MassProperties, eval_body_velocities, eval_fk};

fn properties(mass: f64, com: Vec3, inertia: Mat3) -> MassProperties {
    MassProperties {
        mass,
        local_com: com,
        inertia,
    }
}

fn near(a: Vec3, b: Vec3) {
    assert!((a - b).length() < 1e-10, "{a:?} != {b:?}");
}

#[test]
fn parallel_axis_and_rotated_full_tensor() {
    let one = properties(2.0, -Vec3::X, Mat3::IDENTITY);
    let two = properties(2.0, Vec3::X, Mat3::IDENTITY);
    let sum = one.combined(two).unwrap();
    assert_eq!(sum.mass, 4.0);
    assert_eq!(sum.local_com, Vec3::ZERO);
    assert_eq!(sum.inertia, Mat3::from_diagonal(Vec3::new(2.0, 6.0, 6.0)));
    let rotation = Quat::from_rotation_z(0.7);
    let rotated = sum
        .transformed(Transform::new(Vec3::Y * 3.0, rotation))
        .unwrap();
    near(rotated.local_com, Vec3::Y * 3.0);
    assert!(rotated.inertia.x_axis.y.abs() > 1.0);
    let (moments, axes) = rotated.principal().unwrap();
    let reconstructed = axes * Mat3::from_diagonal(moments) * axes.transpose();
    for i in 0..3 {
        near(reconstructed.col(i), rotated.inertia.col(i));
    }
}

#[test]
fn rejects_invalid_mass_tensors_without_editing_scene() {
    let mut scene = RigidScene::new().unwrap();
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    let stamp = scene.stamp();
    let before = scene.body_mass_properties(body).unwrap();
    for bad in [
        properties(-1.0, Vec3::ZERO, Mat3::IDENTITY),
        properties(f64::INFINITY, Vec3::ZERO, Mat3::IDENTITY),
        properties(0.0, Vec3::ZERO, Mat3::IDENTITY),
        properties(1.0, Vec3::splat(f64::NAN), Mat3::IDENTITY),
        properties(
            1.0,
            Vec3::ZERO,
            Mat3::from_diagonal(Vec3::new(-1.0, 1.0, 1.0)),
        ),
        properties(
            1.0,
            Vec3::ZERO,
            Mat3::from_diagonal(Vec3::new(4.0, 1.0, 1.0)),
        ),
        properties(
            1.0,
            Vec3::ZERO,
            Mat3::from_cols(Vec3::new(1.0, 0.3, 0.0), Vec3::Y, Vec3::Z),
        ),
    ] {
        assert!(scene.set_body_mass_properties(body, bad).is_err());
        assert_eq!(scene.stamp(), stamp);
        assert_eq!(scene.body_mass_properties(body).unwrap(), before);
    }
}

#[test]
fn singular_inertia_is_finite_and_fixed_mass_retains_properties() {
    let mut scene = RigidScene::new().unwrap();
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    let rotation = Mat3::from_quat(Quat::from_rotation_z(0.5));
    let tensor = rotation * Mat3::from_diagonal(Vec3::new(0.0, 2.0, 2.0)) * rotation.transpose();
    let mass = properties(2.0, Vec3::X, tensor);
    scene.set_body_mass_properties(body, mass).unwrap();
    let inverse = scene.model().body_inv_inertia.host().unwrap()[0];
    assert!(inverse.is_finite());
    near(inverse * rotation.x_axis, Vec3::ZERO);
    near(inverse * rotation.y_axis, rotation.y_axis * 0.5);
    scene.set_body_kind(body, BodyKind::Fixed).unwrap();
    assert_eq!(scene.body_mass_properties(body).unwrap(), mass);
    assert_eq!(
        scene.model().body_inv_inertia.host().unwrap()[0],
        Mat3::ZERO
    );
    scene
        .set_body_mass_properties(body, MassProperties::ZERO)
        .unwrap();
    scene.set_body_kind(body, BodyKind::Dynamic).unwrap();
    assert_eq!(
        scene.model().body_inv_inertia.host().unwrap()[0],
        Mat3::ZERO
    );
}

#[test]
fn changing_com_preserves_velocity_field_and_applied_wrench() {
    let mut scene = RigidScene::new().unwrap();
    let mut desc = BodyDesc::default();
    desc.params.pose = Transform::new(Vec3::new(4.0, 5.0, 6.0), Quat::from_rotation_z(0.4));
    desc.velocity = SpatialVector::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.4, 0.5, 0.6));
    let body = scene.insert_body(desc).unwrap();
    let force = SpatialVector::new(Vec3::new(2.0, 3.0, 4.0), Vec3::new(5.0, 6.0, 7.0));
    scene.state_mut().body_f.host_mut().unwrap()[0] = force;
    scene.control_mut().body_external_f.host_mut().unwrap()[0] = force;
    let before = scene.state().body_qd.host().unwrap()[0];
    let pose = scene.state().body_q.host().unwrap()[0];
    let coordinates = scene.state().joint_q.host().unwrap().to_vec();
    let rates = scene.state().joint_qd.host().unwrap().to_vec();
    let com = Vec3::new(0.7, -0.2, 0.3);
    scene
        .set_body_mass_properties(body, properties(3.0, com, Mat3::IDENTITY))
        .unwrap();
    let shift = pose.rotation * com;
    let after = scene.state().body_qd.host().unwrap()[0];
    near(after.linear - after.angular.cross(shift), before.linear);
    near(after.angular, before.angular);
    for wrench in [
        scene.state().body_f.host().unwrap()[0],
        scene.control().body_external_f.host().unwrap()[0],
    ] {
        near(wrench.linear, force.linear);
        near(wrench.angular + shift.cross(wrench.linear), force.angular);
    }
    assert_eq!(scene.state().joint_q.host().unwrap(), coordinates);
    assert_eq!(scene.state().joint_qd.host().unwrap(), rates);
    let mut evaluated = scene.model().state().unwrap();
    evaluated
        .joint_q
        .host_mut()
        .unwrap()
        .copy_from_slice(&coordinates);
    evaluated
        .joint_qd
        .host_mut()
        .unwrap()
        .copy_from_slice(&rates);
    eval_fk(scene.model(), &mut evaluated).unwrap();
    eval_body_velocities(scene.model(), &mut evaluated).unwrap();
    near(evaluated.body_qd.host().unwrap()[0].linear, after.linear);
    near(evaluated.body_qd.host().unwrap()[0].angular, after.angular);
}

#[test]
fn eigendecomposition_is_scale_relative() {
    let rotation = Mat3::from_quat(Quat::from_rotation_z(0.3));
    for scale in [1e-100, 1.0, 1e100] {
        let tensor =
            rotation * Mat3::from_diagonal(Vec3::new(1.0, 2.0, 3.0) * scale) * rotation.transpose();
        let mass = properties(1.0, Vec3::ZERO, tensor);
        let inverse = mass.inverse_inertia().unwrap();
        let identity = tensor * inverse;
        for i in 0..3 {
            near(identity.col(i), Mat3::IDENTITY.col(i));
        }
    }
}

#[test]
fn articulated_com_edits_keep_joint_rates_and_fk_consistent() {
    let mut scene = RigidScene::new().unwrap();
    let parent = scene.insert_body(BodyDesc::default()).unwrap();
    let child = scene.insert_body(BodyDesc::default()).unwrap();
    let joint = scene
        .insert_joint(JointDesc {
            parent: Some(parent),
            child,
            kind: JointType::Revolute,
            frame_parent: Transform::new(Vec3::X, Quat::IDENTITY),
            frame_child: Transform::IDENTITY,
            axis: Vec3::Z,
        })
        .unwrap();
    let joint_index = scene.joint_index(joint).unwrap().index();
    let offset = scene.model().joint_dof_offset.host().unwrap()[joint_index] as usize;
    scene.state_mut().joint_qd.host_mut().unwrap()[offset] = 2.0;
    let coordinates = scene.state().joint_q.host().unwrap().to_vec();
    let rates = scene.state().joint_qd.host().unwrap().to_vec();
    let mut evaluated = scene.model().state().unwrap();
    evaluated
        .joint_q
        .host_mut()
        .unwrap()
        .copy_from_slice(&coordinates);
    evaluated
        .joint_qd
        .host_mut()
        .unwrap()
        .copy_from_slice(&rates);
    eval_fk(scene.model(), &mut evaluated).unwrap();
    eval_body_velocities(scene.model(), &mut evaluated).unwrap();
    scene
        .state_mut()
        .body_q
        .host_mut()
        .unwrap()
        .copy_from_slice(evaluated.body_q.host().unwrap());
    scene
        .state_mut()
        .body_qd
        .host_mut()
        .unwrap()
        .copy_from_slice(evaluated.body_qd.host().unwrap());
    scene
        .set_body_mass_properties(child, properties(2.0, Vec3::X, Mat3::IDENTITY))
        .unwrap();
    eval_fk(scene.model(), &mut evaluated).unwrap();
    eval_body_velocities(scene.model(), &mut evaluated).unwrap();
    assert_eq!(scene.state().joint_qd.host().unwrap(), rates);
    for (actual, expected) in scene
        .state()
        .body_qd
        .host()
        .unwrap()
        .iter()
        .zip(evaluated.body_qd.host().unwrap())
    {
        near(actual.linear, expected.linear);
        near(actual.angular, expected.angular);
    }
}

#[test]
fn com_rebase_overflow_leaves_the_scene_unchanged() {
    let mut scene = RigidScene::new().unwrap();
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    scene.state_mut().body_f.host_mut().unwrap()[0].linear = Vec3::Y * 10.0;
    let stamp = scene.stamp();
    let mass = scene.body_mass_properties(body).unwrap();
    assert!(
        scene
            .set_body_mass_properties(body, properties(1.0, Vec3::X * 1e308, Mat3::IDENTITY))
            .is_err()
    );
    assert_eq!(scene.stamp(), stamp);
    assert_eq!(scene.body_mass_properties(body).unwrap(), mass);
    near(scene.state().body_f.host().unwrap()[0].angular, Vec3::ZERO);
}
