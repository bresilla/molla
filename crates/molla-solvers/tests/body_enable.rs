use molla_math::{Quat, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{
    BodyDesc, BodyHandle, BodyKind, ColliderDesc, ColliderGeometry, JointDesc, JointSettings,
};
use molla_sim::{JointTargetMode, JointType, ShapeFlags};
use molla_solvers::rigid_world::RigidWorld;

fn world() -> RigidWorld {
    let mut world = RigidWorld::new().unwrap();
    world.scene.set_gravity(Vec3::ZERO).unwrap();
    world
}

fn body(world: &mut RigidWorld, x: f64, velocity: f64) -> BodyHandle {
    let mut desc = BodyDesc::default();
    desc.params.pose.position.x = x;
    desc.velocity.linear.x = velocity;
    world.scene.insert_body(desc).unwrap()
}

fn joint(parent: BodyHandle, child: BodyHandle, kind: JointType) -> JointDesc {
    JointDesc {
        parent: Some(parent),
        child,
        kind,
        frame_parent: Transform::IDENTITY,
        frame_child: Transform::IDENTITY,
        axis: Vec3::Z,
    }
}

fn near(a: Vec3, b: Vec3) {
    assert!((a - b).length() < 1e-9, "{a:?} != {b:?}");
}

#[test]
fn disabled_body_keeps_physical_mass_and_pending_motion_but_does_not_move() {
    let mut world = world();
    let b = body(&mut world, 3.0, 2.0);
    let i = world.scene.body_index(b).unwrap().index();
    let mass = world.scene.body_mass_properties(b).unwrap();
    world
        .scene
        .add_body_wrench(b, SpatialVector::new(Vec3::X, Vec3::ZERO))
        .unwrap();
    world.scene.set_body_enabled(b, false).unwrap();
    assert!(!world.scene.body(b).unwrap().enabled);
    assert_eq!(world.scene.body(b).unwrap().kind, BodyKind::Dynamic);
    assert_eq!(world.scene.body_mass_properties(b).unwrap(), mass);
    assert_eq!(world.scene.model().body_inv_mass.host().unwrap()[i], 0.0);
    world.step(0.1).unwrap();
    near(world.scene.body_pose(b).unwrap().position, Vec3::X * 3.0);
    near(world.scene.body_velocity(b).unwrap().linear, Vec3::X * 2.0);
    world.scene.set_body_enabled(b, true).unwrap();
    world.step(0.1).unwrap();
    assert!(world.scene.body_pose(b).unwrap().position.x > 3.2);
    near(world.scene.body_velocity(b).unwrap().linear, Vec3::X * 2.1);
}

#[test]
fn disabled_parent_hides_all_its_colliders_without_changing_authored_flags() {
    let mut world = world();
    let b = body(&mut world, 0.0, 0.0);
    let mut collider = ColliderDesc::new(ColliderGeometry::Ball { radius: 1.0 });
    collider.parent = Some(b);
    collider.sensor = true;
    let shape = world.scene.insert_collider(collider).unwrap();
    let mut other = ColliderDesc::new(ColliderGeometry::Ball { radius: 1.0 });
    other.pose.position.x = 1.0;
    let other = world.scene.insert_collider(other).unwrap();
    world.step(0.01).unwrap();
    assert!(!world.contact_pairs().is_empty());
    let stamp = world.scene.collider_stamp(shape).unwrap();
    world.scene.set_body_enabled(b, false).unwrap();
    assert!(world.scene.collider(shape).unwrap().enabled);
    assert!(!world.scene.collider_is_enabled(shape).unwrap());
    assert_eq!(world.scene.collider_stamp(shape).unwrap(), stamp);
    assert!(world.contact_pairs().is_empty());
    for s in world.scene.collider_indices(shape).unwrap() {
        assert_eq!(
            world.scene.model().shape_flags.host().unwrap()[s.index()] & ShapeFlags::COLLIDES,
            0
        );
    }
    let hit = world
        .cast_ray(Vec3::X * -3.0, Vec3::X, 10.0)
        .unwrap()
        .unwrap();
    assert_eq!(hit.collider, other);
    world.step(0.01).unwrap();
    assert!(world.contact_pairs().is_empty());
    world.scene.set_body_enabled(b, true).unwrap();
    assert!(world.scene.collider_is_enabled(shape).unwrap());
    assert_eq!(
        world
            .cast_ray(Vec3::X * -3.0, Vec3::X, 10.0)
            .unwrap()
            .unwrap()
            .collider,
        shape
    );
}

#[test]
fn disabling_middle_body_splits_joints_without_freezing_healthy_neighbors() {
    let mut world = world();
    let root = body(&mut world, 0.0, 1.0);
    let middle = body(&mut world, 0.0, 1.0);
    let leaf = body(&mut world, 0.0, 1.0);
    let a = world
        .scene
        .insert_joint(joint(root, middle, JointType::Fixed))
        .unwrap();
    let b = world
        .scene
        .insert_joint(joint(middle, leaf, JointType::Fixed))
        .unwrap();
    world.scene.set_body_enabled(middle, false).unwrap();
    assert_eq!(world.scene.joints().count(), 2);
    assert!(world.scene.joint_index(a).is_none());
    assert!(world.scene.joint_index(b).is_none());
    world.step(0.1).unwrap();
    near(world.scene.body_pose(root).unwrap().position, Vec3::X * 0.1);
    near(world.scene.body_pose(leaf).unwrap().position, Vec3::X * 0.1);
    near(world.scene.body_pose(middle).unwrap().position, Vec3::ZERO);
    world
        .scene
        .set_body_pose(middle, Transform::new(Vec3::X * 3.0, Quat::IDENTITY))
        .unwrap();
    near(world.scene.body_pose(leaf).unwrap().position, Vec3::X * 0.1);
    world.scene.set_body_enabled(middle, true).unwrap();
    assert!(world.scene.joint_index(a).is_some());
    assert!(world.scene.joint_index(b).is_some());
    world.step(0.01).unwrap();
    for id in [root, middle, leaf] {
        assert!(world.scene.body_pose(id).unwrap().position.is_finite());
    }
}

#[test]
fn dormant_joint_and_free_root_drives_survive_multiple_rebuilds() {
    let mut world = world();
    let root = body(&mut world, 0.0, 0.0);
    let child = body(&mut world, 0.0, 0.0);
    let hinge = world
        .scene
        .insert_joint(joint(root, child, JointType::Revolute))
        .unwrap();
    let j = world.scene.joint_index(hinge).unwrap().index();
    let offset = world.scene.model().joint_dof_offset.host().unwrap()[j] as usize;
    world
        .scene
        .control_mut()
        .joint_target_mode
        .host_mut()
        .unwrap()[offset] = JointTargetMode::Velocity as u32;
    world
        .scene
        .control_mut()
        .joint_target_vel
        .host_mut()
        .unwrap()[offset] = 2.0;
    world
        .scene
        .control_mut()
        .joint_target_kd
        .host_mut()
        .unwrap()[offset] = 3.0;
    world.scene.control_mut().joint_force.host_mut().unwrap()[0] = 4.0;
    world.scene.set_body_enabled(root, false).unwrap();
    world.scene.set_body_enabled(child, false).unwrap();
    let extra = body(&mut world, 10.0, 0.0);
    world.scene.remove_body(extra).unwrap();
    world.scene.set_body_enabled(child, true).unwrap();
    assert!(world.scene.joint_index(hinge).is_none());
    world.scene.set_body_enabled(root, true).unwrap();
    let j = world.scene.joint_index(hinge).unwrap().index();
    let offset = world.scene.model().joint_dof_offset.host().unwrap()[j] as usize;
    assert_eq!(
        world.scene.control().joint_target_mode.host().unwrap()[offset],
        JointTargetMode::Velocity as u32
    );
    assert_eq!(
        world.scene.control().joint_target_vel.host().unwrap()[offset],
        2.0
    );
    assert_eq!(
        world.scene.control().joint_target_kd.host().unwrap()[offset],
        3.0
    );
    assert_eq!(world.scene.control().joint_force.host().unwrap()[0], 4.0);
}

#[test]
fn world_quarantines_bad_body_state_before_collision_and_steps_healthy_body() {
    let mut world = world();
    let bad = body(&mut world, 0.0, 0.0);
    let good = body(&mut world, 10.0, 1.0);
    let mut shape = ColliderDesc::new(ColliderGeometry::Ball { radius: 1.0 });
    shape.parent = Some(bad);
    let shape = world.scene.insert_collider(shape).unwrap();
    world.scene.state_mut().body_q.host_mut().unwrap()[0]
        .position
        .x = f64::NAN;
    world.scene.state_mut().body_qd.host_mut().unwrap()[0]
        .angular
        .y = f64::NAN;
    world.scene.state_mut().body_f.host_mut().unwrap()[0]
        .linear
        .z = f64::INFINITY;
    world.scene.state_mut().joint_q.host_mut().unwrap()[0] = f64::NAN;
    world
        .scene
        .control_mut()
        .body_external_f
        .host_mut()
        .unwrap()[0]
        .angular
        .x = f64::NAN;
    world.step(0.1).unwrap();
    assert!(!world.scene.body(bad).unwrap().enabled);
    assert_eq!(world.quarantined_bodies(), &[bad]);
    assert!(world.scene.body(good).unwrap().enabled);
    near(
        world.scene.body_pose(good).unwrap().position,
        Vec3::X * 10.1,
    );
    assert!(!world.scene.collider_is_enabled(shape).unwrap());
    assert!(world.scene.body_pose(bad).unwrap().position.is_finite());
    assert_eq!(world.scene.body_velocity(bad).unwrap(), SpatialVector::ZERO);
    assert_eq!(
        world.scene.state().body_f.host().unwrap()[0],
        SpatialVector::ZERO
    );
    let revision = world.scene.revision();
    world.step(0.1).unwrap();
    assert_eq!(world.scene.revision(), revision);
    assert!(world.quarantined_bodies().is_empty());
    world.scene.set_body_enabled(bad, true).unwrap();
    world.step(0.1).unwrap();
    assert!(world.scene.body(bad).unwrap().enabled);
}

#[test]
fn corrupt_motor_state_and_zero_quaternion_are_quarantined_without_revival() {
    let mut world = world();
    let root = body(&mut world, 0.0, 0.0);
    let child = body(&mut world, 0.0, 0.0);
    let hinge = world
        .scene
        .insert_joint(joint(root, child, JointType::Revolute))
        .unwrap();
    let index = world.scene.joint_index(hinge).unwrap().index();
    let offset = world.scene.model().joint_dof_offset.host().unwrap()[index] as usize;
    world
        .scene
        .control_mut()
        .joint_target_kd
        .host_mut()
        .unwrap()[offset] = f64::INFINITY;
    assert_eq!(world.scene.quarantine_non_finite().unwrap(), vec![child]);
    world.scene.set_body_enabled(child, true).unwrap();
    let index = world.scene.joint_index(hinge).unwrap().index();
    let offset = world.scene.model().joint_dof_offset.host().unwrap()[index] as usize;
    assert_eq!(
        world.scene.control().joint_target_kd.host().unwrap()[offset],
        0.0
    );
    world.scene.state_mut().joint_q.host_mut().unwrap()[3..7].fill(0.0);
    assert_eq!(world.scene.quarantine_non_finite().unwrap(), vec![root]);
    world.step(0.01).unwrap();
}

#[test]
fn disabled_authored_graph_still_rejects_cycles_and_invalid_axes_atomically() {
    let mut world = world();
    let a = body(&mut world, 0.0, 0.0);
    let b = body(&mut world, 0.0, 0.0);
    world.scene.set_body_enabled(a, false).unwrap();
    world
        .scene
        .insert_joint(joint(a, b, JointType::Fixed))
        .unwrap();
    let stamp = world.scene.stamp();
    assert!(
        world
            .scene
            .insert_joint(joint(b, a, JointType::Fixed))
            .is_err()
    );
    let mut invalid = joint(b, a, JointType::Revolute);
    invalid.axis = Vec3::ZERO;
    assert!(world.scene.insert_joint(invalid).is_err());
    assert_eq!(world.scene.stamp(), stamp);
}

#[test]
fn authored_disabled_body_accepts_pose_velocity_and_ccd_changes_before_enable() {
    let mut world = world();
    let b = world
        .scene
        .insert_body(BodyDesc {
            enabled: false,
            ..BodyDesc::default()
        })
        .unwrap();
    world
        .scene
        .set_body_pose(b, Transform::new(Vec3::X * 5.0, Quat::IDENTITY))
        .unwrap();
    world
        .scene
        .set_body_velocity(b, SpatialVector::new(Vec3::Y, Vec3::ZERO))
        .unwrap();
    world.scene.set_body_ccd(b, true).unwrap();
    world.step(0.1).unwrap();
    near(world.scene.body_pose(b).unwrap().position, Vec3::X * 5.0);
    world.scene.set_body_enabled(b, true).unwrap();
    assert!(world.scene.body(b).unwrap().ccd_enabled);
    world.step(0.1).unwrap();
    near(
        world.scene.body_pose(b).unwrap().position,
        Vec3::new(5.0, 0.1, 0.0),
    );
}

#[test]
fn joint_disable_preserves_authoring_and_drives_across_body_enable_cycles() {
    let mut world = world();
    let a = body(&mut world, 0.0, 0.0);
    let b = body(&mut world, 0.0, 1.0);
    let hinge = world
        .scene
        .insert_joint(joint(a, b, JointType::Revolute))
        .unwrap();
    let i = world.scene.joint_index(hinge).unwrap().index();
    let offset = world.scene.model().joint_dof_offset.host().unwrap()[i] as usize;
    world
        .scene
        .control_mut()
        .joint_target_vel
        .host_mut()
        .unwrap()[offset] = 7.0;
    world
        .scene
        .set_joint_settings(
            hinge,
            JointSettings {
                enabled: false,
                contacts_enabled: true,
            },
        )
        .unwrap();
    assert!(world.scene.joint_index(hinge).is_none());
    world.scene.set_body_enabled(a, false).unwrap();
    world.scene.set_body_enabled(a, true).unwrap();
    assert!(!world.scene.joint_settings(hinge).unwrap().enabled);
    world.step(0.1).unwrap();
    near(world.scene.body_pose(b).unwrap().position, Vec3::X * 0.1);
    world
        .scene
        .set_joint_settings(hinge, JointSettings::default())
        .unwrap();
    let i = world.scene.joint_index(hinge).unwrap().index();
    let offset = world.scene.model().joint_dof_offset.host().unwrap()[i] as usize;
    assert_eq!(
        world.scene.control().joint_target_vel.host().unwrap()[offset],
        7.0
    );
    world.scene.remove_joint(hinge).unwrap();
    assert!(world.scene.joint_settings(hinge).is_none());
}

#[test]
fn connected_contact_flags_cover_all_colliders_and_remain_authored_when_joint_disabled() {
    let mut world = world();
    let a = body(&mut world, 0.0, 0.0);
    let b = body(&mut world, 0.0, 0.0);
    for parent in [a, b] {
        for x in [-0.1, 0.1] {
            let mut shape = ColliderDesc::new(ColliderGeometry::Ball { radius: 1.0 });
            shape.parent = Some(parent);
            shape.pose.position.x = x;
            shape.sensor = true;
            world.scene.insert_collider(shape).unwrap();
        }
    }
    let j = world
        .scene
        .insert_joint(joint(a, b, JointType::Ball))
        .unwrap();
    world.step(0.01).unwrap();
    assert!(world.contact_pairs().is_empty());
    world
        .scene
        .set_joint_settings(
            j,
            JointSettings {
                enabled: true,
                contacts_enabled: true,
            },
        )
        .unwrap();
    world.step(0.01).unwrap();
    assert_eq!(world.contact_pairs().len(), 4);
    world
        .scene
        .set_joint_settings(
            j,
            JointSettings {
                enabled: false,
                contacts_enabled: false,
            },
        )
        .unwrap();
    world.step(0.01).unwrap();
    assert!(world.contact_pairs().is_empty());
    world.scene.remove_joint(j).unwrap();
    world.step(0.01).unwrap();
    assert_eq!(world.contact_pairs().len(), 4);
}

#[test]
fn local_joint_frames_are_mutable_even_while_suspended_and_invalid_edits_are_atomic() {
    let mut world = world();
    let a = world
        .scene
        .insert_body(BodyDesc {
            kind: BodyKind::Fixed,
            ..BodyDesc::default()
        })
        .unwrap();
    let b = body(&mut world, 0.0, 0.0);
    let j = world
        .scene
        .insert_joint(joint(a, b, JointType::Fixed))
        .unwrap();
    world
        .scene
        .set_joint_settings(
            j,
            JointSettings {
                enabled: false,
                contacts_enabled: false,
            },
        )
        .unwrap();
    let parent = Transform::new(Vec3::Y * 2.0, Quat::IDENTITY);
    let child = Transform::new(Vec3::Y * 0.5, Quat::IDENTITY);
    world.scene.set_joint_frames(j, parent, child).unwrap();
    assert_eq!(world.scene.joint(j).unwrap().frame_parent, parent);
    assert_eq!(world.scene.joint(j).unwrap().frame_child, child);
    let stamp = world.scene.stamp();
    assert!(
        world
            .scene
            .set_joint_frames(
                j,
                Transform::new(Vec3::ZERO, Quat::from_xyzw(0.0, 0.0, 0.0, 0.0)),
                child
            )
            .is_err()
    );
    assert_eq!(world.scene.stamp(), stamp);
    world
        .scene
        .set_joint_settings(j, JointSettings::default())
        .unwrap();
    world.step(0.01).unwrap();
    near(world.scene.body_pose(b).unwrap().position, Vec3::Y * 1.5);
    world
        .scene
        .set_joint_frames(j, parent, Transform::IDENTITY)
        .unwrap();
    world.step(0.01).unwrap();
    near(world.scene.body_pose(b).unwrap().position, Vec3::Y * 2.0);
    world.scene.remove_body(a).unwrap();
    assert!(world.scene.joint(j).is_none());
    assert!(world.scene.joint_settings(j).is_none());
}

#[test]
fn quarantine_restores_latest_committed_pose_not_spawn_pose() {
    let mut world = world();
    let b = body(&mut world, 2.0, 1.0);
    world.step(0.5).unwrap();
    let last = world.scene.body_pose(b).unwrap();
    near(last.position, Vec3::X * 2.5);
    world.scene.state_mut().body_q.host_mut().unwrap()[0]
        .position
        .y = f64::NAN;
    assert_eq!(world.scene.quarantine_non_finite().unwrap(), vec![b]);
    assert_eq!(world.scene.body_pose(b).unwrap(), last);
    world.scene.set_body_enabled(b, true).unwrap();
    world.step(0.5).unwrap();
    near(world.scene.body_pose(b).unwrap().position, Vec3::X * 3.0);
}

#[test]
fn committed_state_validation_is_atomic_and_invalid_dt_does_not_quarantine() {
    let mut world = world();
    let b = body(&mut world, 2.0, 0.0);
    let empty = RigidWorld::new()
        .unwrap()
        .scene
        .state()
        .clone_host()
        .unwrap();
    let stamp = world.scene.stamp();
    assert!(world.scene.commit_step(empty).is_err());
    assert_eq!(world.scene.stamp(), stamp);
    near(world.scene.body_pose(b).unwrap().position, Vec3::X * 2.0);
    world.scene.state_mut().body_f.host_mut().unwrap()[0]
        .linear
        .x = f64::NAN;
    assert!(world.step(f64::NAN).is_err());
    assert!(world.scene.body(b).unwrap().enabled);
    assert_eq!(world.scene.stamp(), stamp);
    world.step(0.01).unwrap();
    assert!(!world.scene.body(b).unwrap().enabled);
}
