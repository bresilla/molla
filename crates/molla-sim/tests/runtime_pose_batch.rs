use molla_math::{Quat, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, BodyHandle, BodyKind, JointDesc, RigidScene};
use molla_sim::{JointType, eval_body_velocities, eval_fk};

fn fixture() -> (RigidScene, [BodyHandle; 4]) {
    let mut scene = RigidScene::new().unwrap();
    let mut bodies = Vec::new();
    for x in [0.0, 1.0, 2.0, 10.0] {
        let mut desc = BodyDesc::default();
        desc.params.pose.position.x = x;
        bodies.push(scene.insert_body(desc).unwrap());
    }
    for (parent, child, kind) in [
        (bodies[0], bodies[1], JointType::Revolute),
        (bodies[1], bodies[2], JointType::Fixed),
    ] {
        scene
            .insert_joint(JointDesc {
                parent: Some(parent),
                child,
                kind,
                frame_parent: Transform::from_translation(Vec3::X),
                frame_child: Transform::IDENTITY,
                axis: Vec3::X,
            })
            .unwrap();
    }
    let mut state = scene.state().clone_host().unwrap();
    for j in 0..scene.model().joint_count {
        let v = scene.model().joint_dof_offset.host().unwrap()[j] as usize;
        let c = scene.model().joint_coord_offset.host().unwrap()[j] as usize;
        if scene.model().joint_type.host().unwrap()[j] == JointType::Revolute as u32 {
            state.joint_q.host_mut().unwrap()[c] = 4.0 * std::f64::consts::TAU + 0.4;
            state.joint_qd.host_mut().unwrap()[v] = 3.0;
        }
    }
    eval_fk(scene.model(), &mut state).unwrap();
    eval_body_velocities(scene.model(), &mut state).unwrap();
    scene.commit_step(state).unwrap();
    (scene, bodies.try_into().unwrap())
}

fn close(a: Transform, b: Transform) {
    assert!((a.position - b.position).length() < 1e-10, "{a:?} != {b:?}");
    assert!(a.rotation.dot(b.rotation).abs() > 1.0 - 1e-12);
}

#[test]
fn group_teleport_is_order_independent_and_preserves_joint_rates_and_winding() {
    let (mut a, ids) = fixture();
    let (mut b, other) = fixture();
    let before: Vec<_> = ids.iter().map(|id| a.body_pose(*id).unwrap()).collect();
    let rates = a.state().joint_qd.host().unwrap().to_vec();
    let coordinates = a.state().joint_q.host().unwrap().to_vec();
    let delta = Transform::new(Vec3::new(5.0, 3.0, -2.0), Quat::from_rotation_y(0.6));
    let edits: Vec<_> = ids[..3]
        .iter()
        .zip(&before)
        .map(|(&id, p)| (id, delta.compose(p)))
        .collect();
    a.set_body_poses(&edits, false).unwrap();
    let reverse: Vec<_> = other[..3]
        .iter()
        .zip(&before)
        .map(|(&id, p)| (id, delta.compose(p)))
        .rev()
        .collect();
    b.set_body_poses(&reverse, false).unwrap();
    for i in 0..4 {
        close(a.body_pose(ids[i]).unwrap(), b.body_pose(other[i]).unwrap());
        close(
            a.body_pose(ids[i]).unwrap(),
            if i < 3 {
                delta.compose(&before[i])
            } else {
                before[i]
            },
        );
    }
    assert_eq!(a.state().joint_qd.host().unwrap(), rates);
    for j in 0..a.model().joint_count {
        if a.model().joint_type.host().unwrap()[j] == JointType::Revolute as u32 {
            let c = a.model().joint_coord_offset.host().unwrap()[j] as usize;
            assert!((a.state().joint_q.host().unwrap()[c] - coordinates[c]).abs() < 1e-12);
        }
    }
}

#[test]
fn root_only_batch_moves_descendants_and_resets_only_its_tree() {
    let (mut scene, ids) = fixture();
    scene
        .set_body_velocity(ids[3], SpatialVector::new(Vec3::Z, Vec3::Y))
        .unwrap();
    let external = scene.body_velocity(ids[3]).unwrap();
    let before: Vec<_> = ids.iter().map(|id| scene.body_pose(*id).unwrap()).collect();
    let delta = Transform::new(Vec3::new(-3.0, 2.0, 7.0), Quat::from_rotation_z(0.3));
    scene
        .set_body_poses(&[(ids[0], delta.compose(&before[0]))], true)
        .unwrap();
    for i in 0..3 {
        close(scene.body_pose(ids[i]).unwrap(), delta.compose(&before[i]));
        assert_eq!(scene.body_velocity(ids[i]).unwrap(), SpatialVector::ZERO);
    }
    assert_eq!(scene.body_velocity(ids[3]).unwrap(), external);
    close(scene.body_pose(ids[3]).unwrap(), before[3]);
}

#[test]
fn invalid_pose_batches_are_atomic() {
    let (mut scene, ids) = fixture();
    let removed = scene.insert_body(BodyDesc::default()).unwrap();
    scene.remove_body(removed).unwrap();
    let revision = scene.revision();
    let stamp = scene.stamp();
    let poses = scene.state().body_q.host().unwrap().to_vec();
    let rates = scene.state().joint_qd.host().unwrap().to_vec();
    let valid = Transform::from_translation(Vec3::splat(10.0));
    for bad in [
        vec![(ids[0], valid), (removed, valid)],
        vec![(ids[0], valid), (ids[0], valid)],
        vec![
            (ids[0], valid),
            (ids[1], Transform::from_translation(Vec3::splat(f64::NAN))),
        ],
    ] {
        assert!(scene.set_body_poses(&bad, true).is_err());
        assert_eq!(scene.revision(), revision);
        assert_eq!(scene.stamp(), stamp);
        assert_eq!(scene.state().body_q.host().unwrap(), poses);
        assert_eq!(scene.state().joint_qd.host().unwrap(), rates);
    }
    scene.set_body_poses(&[], true).unwrap();
    assert_eq!(scene.revision(), revision);
}

#[test]
fn fixed_root_batch_moves_its_dynamic_descendants() {
    let (mut scene, ids) = fixture();
    scene.set_body_kind(ids[0], BodyKind::Fixed).unwrap();
    let before = scene.body_pose(ids[2]).unwrap();
    let delta = Transform::from_translation(Vec3::new(0.0, 4.0, -1.0));
    scene.set_body_poses(&[(ids[0], delta)], true).unwrap();
    close(scene.body_pose(ids[2]).unwrap(), delta.compose(&before));
}
