use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::{
    JointType, MassProperties,
    runtime::{
        BodyDesc, BodyHandle, ColliderDesc, ColliderGeometry, JointBreakLimits, JointBreakTrigger,
        JointDesc, RigidScene,
    },
};

fn body(scene: &mut RigidScene) -> BodyHandle {
    scene
        .insert_body(BodyDesc {
            additional_mass: Some(MassProperties {
                mass: 2.0,
                local_com: Vec3::X * 0.2,
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap()
}

fn joint(parent: Option<BodyHandle>, child: BodyHandle) -> JointDesc {
    JointDesc {
        parent,
        child,
        kind: JointType::Fixed,
        axis: Vec3::X,
        frame_parent: Transform::IDENTITY,
        frame_child: Transform::IDENTITY,
    }
}

#[test]
fn break_thresholds_are_inclusive_independent_and_validate_zero_and_infinity() {
    let limits = JointBreakLimits {
        force: Some(2.0),
        torque: Some(3.0),
    };
    assert_eq!(limits.trigger(1.999, 2.999).unwrap(), None);
    assert_eq!(
        limits.trigger(2.0, 2.999).unwrap(),
        Some(JointBreakTrigger::Force)
    );
    assert_eq!(
        limits.trigger(1.999, 3.0).unwrap(),
        Some(JointBreakTrigger::Torque)
    );
    assert_eq!(
        limits.trigger(2.0, 3.0).unwrap(),
        Some(JointBreakTrigger::Both)
    );
    assert_eq!(
        limits.trigger(20.0, 30.0).unwrap(),
        Some(JointBreakTrigger::Both)
    );
    assert!(
        JointBreakLimits::default()
            .trigger(20.0, 30.0)
            .unwrap()
            .is_none()
    );
    let zero = JointBreakLimits {
        force: Some(0.0),
        torque: None,
    };
    assert_eq!(
        zero.trigger(0.0, 0.0).unwrap(),
        Some(JointBreakTrigger::Force)
    );
    let infinite = JointBreakLimits {
        force: Some(f64::INFINITY),
        torque: Some(f64::INFINITY),
    };
    assert!(!infinite.is_enabled());
    assert!(infinite.trigger(f64::MAX, f64::MAX).unwrap().is_none());
    for value in [-1.0, f64::NEG_INFINITY, f64::NAN] {
        assert!(
            JointBreakLimits {
                force: Some(value),
                torque: None
            }
            .validate()
            .is_err()
        );
        assert!(
            JointBreakLimits {
                force: None,
                torque: Some(value)
            }
            .validate()
            .is_err()
        );
    }
    for value in [-1.0, f64::INFINITY, f64::NAN] {
        assert!(limits.trigger(value, 0.0).is_err());
        assert!(limits.trigger(0.0, value).is_err());
    }
}

#[test]
fn threshold_edits_invalidate_dynamics_without_recompiling_or_changing_state() {
    let mut scene = RigidScene::new().unwrap();
    let b = body(&mut scene);
    let id = scene.insert_joint(joint(None, b)).unwrap();
    let model = scene.model_stamp();
    let dynamics = scene.dynamics_stamp();
    let pose = scene.body_pose(b).unwrap();
    let velocity = scene.body_velocity(b).unwrap();
    let limits = JointBreakLimits {
        force: Some(12.0),
        torque: None,
    };
    scene.set_joint_break_limits(id, limits).unwrap();
    assert_eq!(scene.model_stamp(), model);
    assert_ne!(scene.dynamics_stamp(), dynamics);
    assert_eq!(scene.body_pose(b).unwrap(), pose);
    assert_eq!(scene.body_velocity(b).unwrap(), velocity);
    let dynamics = scene.dynamics_stamp();
    assert!(
        scene
            .set_joint_break_limits(
                id,
                JointBreakLimits {
                    force: Some(-1.0),
                    torque: None
                }
            )
            .is_err()
    );
    assert_eq!(scene.joint_break_limits(id).unwrap(), limits);
    assert_eq!(scene.dynamics_stamp(), dynamics);
    scene.set_joint_break_limits(id, limits).unwrap();
    assert_eq!(scene.dynamics_stamp(), dynamics);
}

#[test]
fn batched_breaks_detach_trees_preserve_state_and_require_explicit_repair() {
    let mut scene = RigidScene::new().unwrap();
    let unrelated = body(&mut scene);
    let a = body(&mut scene);
    let b = body(&mut scene);
    let c = body(&mut scene);
    let ab = scene.insert_joint(joint(Some(a), b)).unwrap();
    let bc = scene.insert_joint(joint(Some(b), c)).unwrap();
    scene
        .set_body_velocity(
            a,
            SpatialVector::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(-0.5, 0.7, 0.2)),
        )
        .unwrap();
    let state: Vec<_> = [a, b, c]
        .into_iter()
        .map(|id| {
            (
                id,
                scene.body_pose(id).unwrap(),
                scene.body_velocity(id).unwrap(),
            )
        })
        .collect();
    let revision = scene.revision();
    assert_eq!(scene.break_joints(&[bc, ab, ab]).unwrap(), vec![ab, bc]);
    assert_eq!(scene.revision(), revision + 1);
    for (id, pose, velocity) in state {
        assert_eq!(scene.body_pose(id).unwrap(), pose);
        assert!((scene.body_velocity(id).unwrap().linear - velocity.linear).length() < 1e-12);
        assert!((scene.body_velocity(id).unwrap().angular - velocity.angular).length() < 1e-12);
    }
    for id in [ab, bc] {
        assert!(scene.joint_broken(id).unwrap());
        assert!(!scene.joint_settings(id).unwrap().enabled);
        assert!(scene.joint_index(id).is_none());
        assert!(scene.joint(id).is_some());
    }
    let stamp = scene.stamp();
    assert!(scene.break_joints(&[ab, bc]).unwrap().is_empty());
    assert_eq!(scene.stamp(), stamp);
    scene.remove_body(unrelated).unwrap();
    assert!(scene.joint_broken(ab).unwrap());
    scene.repair_joint(ab).unwrap();
    assert!(!scene.joint_broken(ab).unwrap());
    assert!(scene.joint_index(ab).is_some());
    assert!(scene.joint_broken(bc).unwrap());
    scene.remove_body(c).unwrap();
    assert!(scene.joint_broken(bc).is_err());
    assert!(scene.joint_break_limits(bc).is_err());
}

#[test]
fn invalid_break_batch_is_atomic_and_loops_do_not_detach_tree_edges() {
    let mut scene = RigidScene::new().unwrap();
    let a = body(&mut scene);
    let b = body(&mut scene);
    let tree = scene.insert_joint(joint(Some(a), b)).unwrap();
    let link = scene
        .insert_rigid_loop_joint(JointDesc {
            kind: JointType::Ball,
            ..joint(None, b)
        })
        .unwrap();
    let removed = scene
        .insert_joint(JointDesc {
            kind: JointType::Distance,
            ..joint(None, a)
        })
        .unwrap();
    scene.remove_joint(removed).unwrap();
    let stamp = scene.stamp();
    assert!(scene.break_joints(&[tree, removed]).is_err());
    assert_eq!(scene.stamp(), stamp);
    assert!(!scene.joint_broken(tree).unwrap());
    scene.break_joints(&[link]).unwrap();
    assert!(scene.joint_index(tree).is_some());
    assert!(!scene.joint_broken(tree).unwrap());
    assert!(scene.joint_broken(link).unwrap());
    let mut settings = scene.joint_settings(link).unwrap();
    settings.enabled = true;
    scene.set_joint_settings(link, settings).unwrap();
    assert!(!scene.joint_broken(link).unwrap());
}

#[test]
fn broken_tree_edges_release_authored_topology_and_conflicting_repair_is_atomic() {
    for anchored in [false, true] {
        let mut scene = RigidScene::new().unwrap();
        let a = body(&mut scene);
        let b = body(&mut scene);
        let parent = if anchored { None } else { Some(a) };
        let broken = scene.insert_joint(joint(parent, b)).unwrap();
        scene.break_joints(&[broken]).unwrap();
        let replacement = scene.insert_joint(joint(parent, b)).unwrap();
        assert!(scene.joint_index(replacement).is_some());
        let stamp = scene.stamp();
        assert!(scene.repair_joint(broken).is_err());
        assert_eq!(scene.stamp(), stamp);
        assert!(scene.joint_broken(broken).unwrap());
        assert!(scene.joint_index(replacement).is_some());
        scene.remove_joint(replacement).unwrap();
        scene.repair_joint(broken).unwrap();
        assert!(!scene.joint_broken(broken).unwrap());
        assert!(scene.joint_index(broken).is_some());
    }
}

#[test]
fn break_removes_only_joint_collision_policy_and_explicit_pairs_survive_compaction() {
    let mut scene = RigidScene::new().unwrap();
    let removed = body(&mut scene);
    let a = body(&mut scene);
    let b = body(&mut scene);
    let collider = |parent| ColliderDesc {
        parent: Some(parent),
        ..ColliderDesc::new(ColliderGeometry::Ball { radius: 0.1 })
    };
    let ca = scene.insert_collider(collider(a)).unwrap();
    let cb = scene.insert_collider(collider(b)).unwrap();
    let id = scene.insert_joint(joint(Some(a), b)).unwrap();
    assert_eq!(scene.model().shape_collision_filter_pairs.len(), 1);
    scene.break_joints(&[id]).unwrap();
    assert!(scene.model().shape_collision_filter_pairs.is_empty());
    scene.repair_joint(id).unwrap();
    scene.set_body_pair_excluded(b, a, true).unwrap();
    scene.break_joints(&[id]).unwrap();
    assert_eq!(scene.model().shape_collision_filter_pairs.len(), 1);
    assert!(scene.body_pair_excluded(a, b).unwrap());
    scene.set_collider_pair_excluded(cb, ca, true).unwrap();
    scene.set_body_pair_excluded(a, b, false).unwrap();
    assert_eq!(scene.model().shape_collision_filter_pairs.len(), 1);
    scene.remove_body(removed).unwrap();
    assert_eq!(scene.model().shape_collision_filter_pairs.len(), 1);
    assert!(scene.collider_pair_excluded(ca, cb).unwrap());
    scene.set_collider_pair_excluded(ca, cb, false).unwrap();
    assert!(scene.model().shape_collision_filter_pairs.is_empty());
    scene.set_collider_pair_excluded(ca, cb, true).unwrap();
    scene.remove_collider(ca).unwrap();
    assert!(scene.model().shape_collision_filter_pairs.is_empty());
    assert!(scene.collider_pair_excluded(ca, cb).is_err());
}

#[test]
fn explicit_filters_expand_compound_colliders_and_future_body_shapes() {
    let mut scene = RigidScene::new().unwrap();
    let a = body(&mut scene);
    let b = body(&mut scene);
    let unused = scene
        .insert_collider(ColliderDesc::new(ColliderGeometry::Ball { radius: 0.1 }))
        .unwrap();
    let ca = scene
        .insert_collider(ColliderDesc {
            parent: Some(a),
            ..ColliderDesc::new(ColliderGeometry::Ball { radius: 0.1 })
        })
        .unwrap();
    let tetrahedron = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
    let cb = scene
        .insert_collider(ColliderDesc {
            parent: Some(b),
            ..ColliderDesc::new(ColliderGeometry::ConvexParts(vec![
                tetrahedron.clone(),
                tetrahedron,
            ]))
        })
        .unwrap();
    assert_eq!(scene.collider_indices(cb).unwrap().len(), 2);
    scene.set_body_pair_excluded(a, b, true).unwrap();
    assert_eq!(scene.model().shape_collision_filter_pairs.len(), 2);
    scene
        .insert_collider(ColliderDesc {
            parent: Some(a),
            ..ColliderDesc::new(ColliderGeometry::Ball { radius: 0.2 })
        })
        .unwrap();
    assert_eq!(scene.model().shape_collision_filter_pairs.len(), 4);
    scene.set_collider_pair_excluded(ca, cb, true).unwrap();
    scene.set_body_pair_excluded(a, b, false).unwrap();
    assert_eq!(scene.model().shape_collision_filter_pairs.len(), 2);
    scene.remove_collider(unused).unwrap();
    let a = scene.collider_indices(ca).unwrap()[0].index() as u32;
    for b in scene.collider_indices(cb).unwrap() {
        let b = b.index() as u32;
        assert!(
            scene
                .model()
                .shape_collision_filter_pairs
                .contains(&(a.min(b), a.max(b)))
        );
    }
}

#[test]
fn broken_edges_do_not_propagate_later_structural_or_state_edits() {
    let mut scene = RigidScene::new().unwrap();
    let a = body(&mut scene);
    let b = body(&mut scene);
    let id = scene.insert_joint(joint(Some(a), b)).unwrap();
    scene.break_joints(&[id]).unwrap();
    let structure = scene.body_structure_stamp(b).unwrap();
    scene
        .set_body_mass_properties(
            a,
            MassProperties {
                mass: 3.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            },
        )
        .unwrap();
    assert_eq!(scene.body_structure_stamp(b).unwrap(), structure);
    let state = scene.body_state_stamp(b).unwrap();
    scene
        .set_body_velocity(a, SpatialVector::new(Vec3::X, Vec3::ZERO))
        .unwrap();
    assert_eq!(scene.body_state_stamp(b).unwrap(), state);
}

#[test]
fn step_checkpoint_restores_solver_state_and_forks_do_not_share_mutable_dynamics() {
    let mut scene = RigidScene::new().unwrap();
    let a = body(&mut scene);
    scene
        .set_body_velocity(a, SpatialVector::new(Vec3::X, Vec3::Y))
        .unwrap();
    let checkpoint = scene.checkpoint_step().unwrap();
    let dynamics = scene.dynamics_stamp();
    let activation = scene.activation_stamp(a).unwrap();
    let pose = scene.body_pose(a).unwrap();
    let velocity = scene.body_velocity(a).unwrap();
    let mut next = scene.state().clone_host().unwrap();
    next.body_q.host_mut().unwrap()[0].position += Vec3::Z;
    next.body_qd.host_mut().unwrap()[0].linear *= 2.0;
    scene.commit_step(next).unwrap();
    scene.restore_step(checkpoint).unwrap();
    assert_eq!(scene.body_pose(a).unwrap(), pose);
    assert_eq!(scene.body_velocity(a).unwrap(), velocity);
    assert_eq!(scene.dynamics_stamp(), dynamics);
    assert_eq!(scene.activation_stamp(a).unwrap(), activation);
    let mut fork = scene.fork().unwrap();
    assert_ne!(fork.model_stamp(), scene.model_stamp());
    assert_eq!(fork.body_pose(a).unwrap(), pose);
    assert_eq!(fork.body_velocity(a).unwrap(), velocity);
    fork.set_body_velocity(a, SpatialVector::ZERO).unwrap();
    assert_eq!(scene.body_velocity(a).unwrap(), velocity);
    let checkpoint = scene.checkpoint_step().unwrap();
    scene.set_gravity(Vec3::X).unwrap();
    assert!(scene.restore_step(checkpoint).is_err());
    assert_eq!(scene.gravity(), Vec3::X);
    let checkpoint = scene.checkpoint_step().unwrap();
    scene.control_mut();
    assert!(scene.restore_step(checkpoint).is_err());
    let checkpoint = scene.checkpoint_step().unwrap();
    scene.insert_joint(joint(None, a)).unwrap();
    assert!(scene.restore_step(checkpoint).is_err());
}
