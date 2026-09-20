use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxis, JointMotor, JointSoftness};
use molla_sim::runtime::{
    BodyDesc, BodyHandle, BodyKind, ColliderDesc, ColliderGeometry, JointDesc,
};
use molla_sim::{BodyParams, JointTargetMode, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

fn body(world: &mut RigidWorld, x: f64, kind: BodyKind) -> BodyHandle {
    world
        .scene
        .insert_body(BodyDesc {
            kind,
            params: BodyParams::new().with_position(Vec3::new(x, 2.0, 0.0)),
            additional_mass: Some(MassProperties {
                mass: 1.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap()
}

fn sphere(world: &mut RigidWorld, body: BodyHandle, sensor: bool) {
    let mut collider = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
    collider.parent = Some(body);
    collider.sensor = sensor;
    world.scene.insert_collider(collider).unwrap();
}

fn two() -> (RigidWorld, BodyHandle, BodyHandle) {
    let mut world = RigidWorld::new().unwrap();
    let a = body(&mut world, 0.0, BodyKind::Dynamic);
    let b = body(&mut world, 3.0, BodyKind::Dynamic);
    (world, a, b)
}

#[test]
fn explicit_sleep_skips_gravity_and_preserves_independent_groups() {
    let (mut world, a, b) = two();
    world
        .scene
        .set_body_velocity(a, SpatialVector::new(Vec3::X, Vec3::Y))
        .unwrap();
    world.sleep_body(a).unwrap();
    let pose = world.scene.body_pose(a).unwrap();
    assert_eq!(world.scene.body_velocity(a).unwrap(), SpatialVector::ZERO);
    assert!(world.is_sleeping(a));
    assert!(!world.is_sleeping(b));
    for _ in 0..100 {
        world.step(0.001).unwrap();
    }
    assert_eq!(world.scene.body_pose(a).unwrap(), pose);
    assert_eq!(world.scene.body_velocity(a).unwrap(), SpatialVector::ZERO);
    assert!(world.scene.body_pose(b).unwrap().position.y < pose.position.y);
    world.sleep_body(b).unwrap();
    assert!(world.is_sleeping(a) && world.is_sleeping(b));
    world.step(0.001).unwrap();
    world.wake_body(a).unwrap();
    assert!(!world.is_sleeping(a) && world.is_sleeping(b));
    world.step(0.001).unwrap();
    assert!(world.scene.body_velocity(a).unwrap().linear.y < 0.0);
    assert_eq!(world.scene.body_velocity(b).unwrap(), SpatialVector::ZERO);
}

#[test]
fn body_edits_wake_only_their_dependency_group_and_invalid_edits_are_atomic() {
    let (mut world, a, b) = two();
    for edit in 0..5 {
        world.sleep_body(a).unwrap();
        world.sleep_body(b).unwrap();
        assert!(
            world
                .scene
                .set_body_velocity(a, SpatialVector::new(Vec3::splat(f64::NAN), Vec3::ZERO))
                .is_err()
        );
        assert!(world.is_sleeping(a) && world.is_sleeping(b));
        match edit {
            0 => world
                .scene
                .set_body_velocity(a, SpatialVector::new(Vec3::X, Vec3::ZERO))
                .unwrap(),
            1 => world
                .scene
                .set_body_pose(a, Transform::from_translation(Vec3::Y * 3.0))
                .unwrap(),
            2 => world
                .scene
                .add_body_wrench(a, SpatialVector::new(Vec3::X, Vec3::ZERO))
                .unwrap(),
            3 => world.scene.reset_body_forces(a).unwrap(),
            _ => world.scene.set_body_damping(a, 0.1, 0.2).unwrap(),
        }
        assert!(!world.is_sleeping(a) && world.is_sleeping(b), "edit={edit}");
        world.step(0.001).unwrap();
        assert!(!world.is_sleeping(a) && world.is_sleeping(b));
    }
}

#[test]
fn global_and_raw_mutations_wake_sleepers_but_noop_gravity_does_not() {
    let (mut world, a, b) = two();
    for edit in 0..4 {
        world.sleep_body(a).unwrap();
        world.sleep_body(b).unwrap();
        world.scene.set_gravity(world.scene.gravity()).unwrap();
        assert!(world.is_sleeping(a) && world.is_sleeping(b));
        match edit {
            0 => world.scene.set_gravity(Vec3::ZERO).unwrap(),
            1 => {
                let _ = world.scene.state_mut();
            }
            2 => {
                let _ = world.scene.control_mut();
            }
            _ => world
                .scene
                .commit_step(world.scene.state().clone_host().unwrap())
                .unwrap(),
        }
        assert!(!world.is_sleeping(a) && !world.is_sleeping(b));
    }
}

#[test]
fn fresh_substep_contact_wakes_sleeping_body_but_filtered_or_sensor_contacts_do_not() {
    for mode in 0..3 {
        let (mut world, a, b) = two();
        world.scene.set_gravity(Vec3::ZERO).unwrap();
        world
            .scene
            .set_body_pose(b, Transform::from_translation(Vec3::new(1.4, 2.0, 0.0)))
            .unwrap();
        sphere(&mut world, a, false);
        sphere(&mut world, b, mode == 1);
        world
            .scene
            .set_body_velocity(b, SpatialVector::new(-Vec3::X * 20.0, Vec3::ZERO))
            .unwrap();
        world.sleep_body(a).unwrap();
        let before = world.scene.body_pose(a).unwrap();
        world.step_filtered(0.04, |_, _| mode == 2).unwrap();
        assert_eq!(world.is_sleeping(a), mode != 0);
        if mode == 0 {
            assert!(world.scene.body_velocity(a).unwrap().linear.x < -0.1);
            assert!(
                world
                    .contact_pairs()
                    .iter()
                    .flat_map(|p| &p.points)
                    .any(|p| p.acted)
            );
        } else {
            assert_eq!(world.scene.body_pose(a).unwrap(), before);
        }
    }
}

#[test]
fn moving_kinematic_contact_wakes_a_previously_unconnected_sleeper() {
    let (mut world, a, b) = two();
    world.scene.set_body_kind(b, BodyKind::Kinematic).unwrap();
    world
        .scene
        .set_body_pose(b, Transform::from_translation(Vec3::new(0.9, 2.0, 0.0)))
        .unwrap();
    sphere(&mut world, a, false);
    sphere(&mut world, b, false);
    world
        .scene
        .set_body_velocity(b, SpatialVector::new(-Vec3::X, Vec3::ZERO))
        .unwrap();
    world.sleep_body(a).unwrap();
    assert!(world.is_sleeping(a));
    world.step(0.001).unwrap();
    assert!(!world.is_sleeping(a));
    assert!(world.scene.body_velocity(a).unwrap().linear.x < 0.0);
}

#[test]
fn loop_coupling_and_motor_changes_wake_the_complete_group() {
    let (mut world, a, b) = two();
    let joint = world
        .scene
        .insert_loop_joint(
            JointDesc {
                parent: Some(a),
                child: b,
                kind: JointType::Revolute,
                frame_parent: Transform::from_translation(Vec3::X * 3.0),
                frame_child: Transform::IDENTITY,
                axis: Vec3::Z,
            },
            JointSoftness {
                natural_frequency: 20.0,
                damping_ratio: 1.0,
            },
        )
        .unwrap();
    let motor = JointMotor {
        mode: JointTargetMode::Velocity,
        target_velocity: 1.0,
        damping: 2.0,
        ..Default::default()
    };
    world
        .scene
        .set_joint_motor(joint, JointAxis::AngX, motor)
        .unwrap();
    world.sleep_body(a).unwrap();
    assert!(world.is_sleeping(a) && world.is_sleeping(b));
    world
        .scene
        .set_joint_motor(joint, JointAxis::AngX, motor)
        .unwrap();
    assert!(world.is_sleeping(a) && world.is_sleeping(b));
    world.step(0.001).unwrap();
    assert!(world.is_sleeping(a) && world.is_sleeping(b));
    world
        .scene
        .set_joint_motor(
            joint,
            JointAxis::AngX,
            JointMotor {
                target_velocity: 2.0,
                ..motor
            },
        )
        .unwrap();
    assert!(!world.is_sleeping(a) && !world.is_sleeping(b));
    world.step(0.001).unwrap();
    assert!(world.scene.body_velocity(b).unwrap().angular.length() > 0.0);
}

#[test]
fn tree_motor_restoring_a_raw_override_counts_as_a_change() {
    let (mut world, a, b) = two();
    let joint = world
        .scene
        .insert_joint(JointDesc {
            parent: Some(a),
            child: b,
            kind: JointType::Revolute,
            frame_parent: Transform::from_translation(Vec3::X * 3.0),
            frame_child: Transform::IDENTITY,
            axis: Vec3::Z,
        })
        .unwrap();
    let motor = JointMotor {
        mode: JointTargetMode::Velocity,
        target_velocity: 1.0,
        damping: 2.0,
        ..Default::default()
    };
    world
        .scene
        .set_joint_motor(joint, JointAxis::AngX, motor)
        .unwrap();
    let index = world.scene.joint_index(joint).unwrap().index();
    let dof = world.scene.model().joint_dof_offset.host().unwrap()[index] as usize;
    world
        .scene
        .control_mut()
        .joint_target_vel
        .host_mut()
        .unwrap()[dof] = 2.0;
    world.sleep_body(a).unwrap();
    world
        .scene
        .set_joint_motor(joint, JointAxis::AngX, motor)
        .unwrap();
    assert!(!world.is_sleeping(a) && !world.is_sleeping(b));
}

#[test]
fn failed_steps_preserve_sleep_and_custom_forces_wake_conservatively_on_success() {
    let (mut world, a, b) = two();
    world.sleep_body(a).unwrap();
    world.sleep_body(b).unwrap();
    let pose = world.scene.body_pose(a).unwrap();
    assert!(world.step(0.0).is_err());
    assert!(
        world
            .step_with_forces(
                0.001,
                |_, _| false,
                |_, _, _, _| Err(molla_core::Error::Solver("injected".into()))
            )
            .is_err()
    );
    assert!(world.is_sleeping(a) && world.is_sleeping(b));
    assert_eq!(world.scene.body_pose(a).unwrap(), pose);
    world
        .step_with_forces(0.001, |_, _| false, |_, _, _, _| Ok(()))
        .unwrap();
    assert!(!world.is_sleeping(a) && !world.is_sleeping(b));
    assert!(world.scene.body_velocity(a).unwrap().linear.y < 0.0);
}

#[test]
fn topology_edits_invalidate_sleep_without_stale_handle_aliasing() {
    let (mut world, a, b) = two();
    world.sleep_body(a).unwrap();
    world.sleep_body(b).unwrap();
    world.scene.remove_body(a).unwrap();
    let c = body(&mut world, 6.0, BodyKind::Dynamic);
    assert!(!world.is_sleeping(a) && !world.is_sleeping(b) && !world.is_sleeping(c));
    assert!(world.sleep_body(a).is_err());
    assert!(world.wake_body(a).is_err());
    world.sleep_body(c).unwrap();
    world.step(0.001).unwrap();
    assert!(world.is_sleeping(c));
    assert!(world.scene.body_velocity(b).unwrap().linear.y < 0.0);
}

#[test]
fn replacing_a_scene_with_matching_handles_and_edit_counts_does_not_reuse_sleep() {
    let (mut world, a, _) = two();
    world.sleep_body(a).unwrap();
    let (mut replacement, replacement_a, _) = two();
    assert_eq!(a, replacement_a);
    replacement
        .scene
        .commit_step(replacement.scene.state().clone_host().unwrap())
        .unwrap();
    world.scene = replacement.scene;
    assert!(!world.is_sleeping(a));
    world.step(0.001).unwrap();
    assert!(world.scene.body_velocity(a).unwrap().linear.y < 0.0);
}

#[test]
fn a_sleeping_remote_body_does_not_change_active_damping_or_motion() {
    let (mut selected, a, b) = two();
    let (mut ordinary, reference, _) = two();
    for (world, body) in [(&mut selected, a), (&mut ordinary, reference)] {
        world
            .scene
            .set_body_velocity(body, SpatialVector::new(Vec3::X, Vec3::Z))
            .unwrap();
        world.scene.set_body_damping(body, 0.15, 0.3).unwrap();
    }
    selected.sleep_body(b).unwrap();
    for _ in 0..100 {
        selected.step(0.001).unwrap();
        ordinary.step(0.001).unwrap();
        assert_eq!(
            selected.scene.body_pose(a).unwrap(),
            ordinary.scene.body_pose(reference).unwrap()
        );
        assert_eq!(
            selected.scene.body_velocity(a).unwrap(),
            ordinary.scene.body_velocity(reference).unwrap()
        );
    }
}

#[test]
fn sleeping_contacts_report_no_impulse_and_static_boundary_edits_wake() {
    let (mut world, a, ground) = two();
    world.scene.set_body_kind(ground, BodyKind::Fixed).unwrap();
    world
        .scene
        .set_body_pose(
            ground,
            Transform::from_translation(Vec3::new(0.9, 2.0, 0.0)),
        )
        .unwrap();
    sphere(&mut world, a, false);
    sphere(&mut world, ground, false);
    world.step(0.0001).unwrap();
    world.sleep_body(a).unwrap();
    world.step(0.001).unwrap();
    assert!(world.is_sleeping(a));
    let pairs = world.contact_pairs();
    assert!(!pairs.is_empty());
    assert!(
        pairs
            .iter()
            .flat_map(|pair| &pair.points)
            .all(|point| !point.acted && point.normal_impulse == 0.0)
    );
    world
        .scene
        .set_body_pose(
            ground,
            Transform::from_translation(Vec3::new(0.8, 2.0, 0.0)),
        )
        .unwrap();
    assert!(!world.is_sleeping(a));
    world.step(0.001).unwrap();
    assert!(world.scene.body_velocity(a).unwrap().linear.x < 0.0);
}

#[test]
fn changed_pair_filters_wake_on_lost_support_and_new_contacts() {
    for initially_excluded in [false, true] {
        let (mut world, a, b) = two();
        world.scene.set_body_kind(b, BodyKind::Fixed).unwrap();
        world
            .scene
            .set_body_pose(b, Transform::from_translation(Vec3::new(0.9, 2.0, 0.0)))
            .unwrap();
        sphere(&mut world, a, false);
        sphere(&mut world, b, false);
        world
            .step_filtered(0.0001, |_, _| initially_excluded)
            .unwrap();
        world.sleep_body(a).unwrap();
        assert!(world.is_sleeping(a));
        world
            .step_filtered(0.001, |_, _| !initially_excluded)
            .unwrap();
        assert!(!world.is_sleeping(a));
    }
}

#[test]
#[ignore = "CPU explicit-sleep fleet benchmark; run without concurrent builds"]
fn explicit_sleep_fleet_timings() {
    for count in [1, 32, 128] {
        let mut world = RigidWorld::new().unwrap();
        world.scene.set_gravity(Vec3::ZERO).unwrap();
        let bodies: Vec<_> = (0..count)
            .map(|i| body(&mut world, i as f64 * 3.0, BodyKind::Dynamic))
            .collect();
        for &body in &bodies {
            world.sleep_body(body).unwrap();
        }
        for active in [0, 1, count] {
            for &body in &bodies[..active] {
                world.wake_body(body).unwrap();
            }
            let mut samples = Vec::new();
            for i in 0..100 {
                let start = std::time::Instant::now();
                world.step(1.0 / 120.0).unwrap();
                if i >= 20 {
                    samples.push(start.elapsed().as_secs_f64() * 1000.0);
                }
            }
            assert_eq!(
                bodies
                    .iter()
                    .filter(|&&body| world.is_sleeping(body))
                    .count(),
                count - active
            );
            samples.sort_by(f64::total_cmp);
            eprintln!(
                "explicit sleep: bodies={count} awake={active} median_ms={:.6} p95_ms={:.6}",
                samples[40], samples[76]
            );
        }
    }
}
