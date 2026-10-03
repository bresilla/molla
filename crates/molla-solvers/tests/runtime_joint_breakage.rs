use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::{
    JointType, MassProperties,
    runtime::{BodyDesc, BodyHandle, JointBreakLimits, JointBreakTrigger, JointDesc, JointHandle},
};
use molla_solvers::rigid_world::RigidWorld;

fn fixed(world: &mut RigidWorld) -> (BodyHandle, JointHandle) {
    let child = world
        .scene
        .insert_body(BodyDesc {
            additional_mass: Some(MassProperties {
                mass: 2.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * 4.0,
            }),
            ..Default::default()
        })
        .unwrap();
    let joint = world
        .scene
        .insert_joint(JointDesc {
            parent: None,
            child,
            kind: JointType::Fixed,
            axis: Vec3::Z,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    (child, joint)
}
fn limits(force: Option<f64>, torque: Option<f64>) -> JointBreakLimits {
    JointBreakLimits { force, torque }
}
fn world() -> RigidWorld {
    let mut world = RigidWorld::new().unwrap();
    world.scene.set_gravity(Vec3::ZERO).unwrap();
    world.set_reaction_readout(true).unwrap();
    world
}

#[test]
fn gravity_break_occurs_before_the_next_substep_not_at_frame_end() {
    for substeps in [1, 4, 8] {
        for dt in [0.04, 0.02] {
            let mut world = world();
            let (body, joint) = fixed(&mut world);
            world.scene.set_gravity(-Vec3::Y * 10.0).unwrap();
            world.solver.substeps = substeps;
            world
                .scene
                .set_joint_break_limits(joint, limits(Some(20.0), None))
                .unwrap();
            world.step(dt).unwrap();
            assert!(world.scene.joint_broken(joint).unwrap());
            let events = world.joint_break_events();
            assert_eq!(events.len(), 1);
            let event = &events[0];
            assert_eq!(event.trigger, JointBreakTrigger::Force);
            assert_eq!(event.sample.substep, 0);
            assert_eq!(event.sample.time_offset, dt / substeps as f64);
            assert!((event.sample.child.linear.y - 20.0).abs() < 1e-9);
            let free = (substeps - 1) as f64;
            let h = dt / substeps as f64;
            assert!(
                (world.scene.body_velocity(body).unwrap().linear.y + free * h * 10.0).abs() < 1e-10
            );
            assert!(
                (world.scene.body_pose(body).unwrap().position.y
                    + free * (free + 1.0) * 0.5 * h * h * 10.0)
                    .abs()
                    < 1e-10
            );
            assert!(
                (world.joint_reaction(joint).unwrap().child.linear.y - 20.0 / substeps as f64)
                    .abs()
                    < 1e-9
            );
            world.step(dt).unwrap();
            assert_eq!(world.joint_break_events().len(), 1);
        }
    }
}

#[test]
fn transient_peak_triggers_below_frame_mean_threshold_and_repair_starts_a_new_episode() {
    let mut world = world();
    let (body, joint) = fixed(&mut world);
    world.solver.substeps = 4;
    world
        .scene
        .set_joint_break_limits(joint, limits(Some(5.0), None))
        .unwrap();
    let mut calls = 0;
    world
        .step_with_forces(
            0.04,
            |_, _| false,
            |scene, _, h, forces| {
                assert_eq!(h, 0.01);
                forces[scene.body_index(body).unwrap().index()].linear.x +=
                    [0.0, 8.0, 0.0, 0.0][calls];
                calls += 1;
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(calls, 4);
    assert_eq!(world.joint_break_events().len(), 1);
    assert_eq!(world.joint_break_events()[0].sample.substep, 1);
    assert_eq!(world.joint_break_events()[0].sample.time_offset, 0.02);
    let reading = world.joint_reaction(joint).unwrap();
    assert_eq!(reading.peak_force, 8.0);
    assert_eq!(reading.child.linear.x, -2.0);
    assert_eq!(reading.sample_dt, 0.02);
    world.scene.repair_joint(joint).unwrap();
    world
        .scene
        .set_joint_break_limits(joint, limits(Some(0.0), None))
        .unwrap();
    world.step(0.04).unwrap();
    assert_eq!(world.joint_break_events().len(), 2);
    assert_eq!(world.drain_joint_break_events().len(), 2);
    assert!(world.joint_break_events().is_empty());
    world.step(0.04).unwrap();
    assert!(world.joint_break_events().is_empty());
}

#[test]
fn channels_simultaneous_breaks_and_disabled_limits_follow_actual_loads() {
    let mut world = world();
    world.solver.substeps = 4;
    let mut joints = Vec::new();
    for threshold in [
        limits(Some(6.0), Some(5.0)),
        limits(Some(7.0), Some(4.0)),
        limits(Some(6.0), Some(4.0)),
        limits(Some(6.1), Some(4.1)),
        limits(None, None),
        limits(Some(f64::INFINITY), Some(f64::INFINITY)),
    ] {
        let (body, joint) = fixed(&mut world);
        world
            .scene
            .add_body_wrench(body, SpatialVector::new(Vec3::X * 6.0, Vec3::Z * 4.0))
            .unwrap();
        world
            .scene
            .set_joint_break_limits(joint, threshold)
            .unwrap();
        joints.push(joint);
    }
    let revision = world.scene.revision();
    world.step(0.01).unwrap();
    assert_eq!(world.scene.revision(), revision + 1);
    assert_eq!(
        world
            .joint_break_events()
            .iter()
            .map(|e| e.trigger)
            .collect::<Vec<_>>(),
        vec![
            JointBreakTrigger::Force,
            JointBreakTrigger::Torque,
            JointBreakTrigger::Both
        ]
    );
    for (i, joint) in joints.into_iter().enumerate() {
        assert_eq!(world.scene.joint_broken(joint).unwrap(), i < 3);
    }
}

#[test]
fn later_force_failure_rolls_back_state_topology_events_and_input_validity() {
    for fail_on in [0, 2] {
        let mut world = world();
        let (body, joint) = fixed(&mut world);
        world.solver.substeps = 4;
        world
            .scene
            .set_joint_break_limits(joint, limits(Some(1.0), None))
            .unwrap();
        let stamp = world.scene.stamp();
        let model = world.scene.model_stamp();
        let dynamics = world.scene.dynamics_stamp();
        let pose = world.scene.body_pose(body).unwrap();
        let velocity = world.scene.body_velocity(body).unwrap();
        let activation = world.scene.activation_stamp(body).unwrap();
        let revision = world.scene.revision();
        let mut calls = 0;
        assert!(
            world
                .step_with_forces(
                    0.04,
                    |_, _| false,
                    |scene, _, _, forces| {
                        let i = calls;
                        calls += 1;
                        if i == fail_on {
                            return Err(molla_core::Error::Solver(
                                "injected post-break failure".into(),
                            ));
                        }
                        forces[scene.body_index(body).unwrap().index()].linear.x += 2.0;
                        Ok(())
                    }
                )
                .is_err()
        );
        assert_eq!(world.scene.stamp(), stamp);
        assert_eq!(world.scene.model_stamp(), model);
        assert_eq!(world.scene.dynamics_stamp(), dynamics);
        assert_eq!(world.scene.activation_stamp(body).unwrap(), activation);
        assert_eq!(world.scene.revision(), revision);
        assert_eq!(world.scene.body_pose(body).unwrap(), pose);
        assert_eq!(world.scene.body_velocity(body).unwrap(), velocity);
        assert!(!world.scene.joint_broken(joint).unwrap());
        assert!(world.joint_break_events().is_empty());
        assert!(world.joint_reaction_samples().is_none());
        assert_eq!(world.solver.substeps, 4);
        world
            .scene
            .add_body_wrench(body, SpatialVector::new(Vec3::X * 2.0, Vec3::ZERO))
            .unwrap();
        world.step(0.04).unwrap();
        assert!(world.scene.joint_broken(joint).unwrap());
        assert_eq!(world.joint_break_events().len(), 1);
    }
}

#[test]
fn sleeping_zero_threshold_joint_waits_for_a_valid_sample_and_readout_is_optional() {
    let mut world = world();
    let (body, joint) = fixed(&mut world);
    let (other, _) = fixed(&mut world);
    world
        .scene
        .set_joint_break_limits(joint, limits(Some(0.0), None))
        .unwrap();
    world.sleep_body(body).unwrap();
    world.sleep_body(other).unwrap();
    world.set_reaction_readout(false).unwrap();
    world.step(0.01).unwrap();
    assert!(!world.scene.joint_broken(joint).unwrap());
    assert!(world.joint_break_events().is_empty());
    world.wake_body(body).unwrap();
    world.step(0.01).unwrap();
    assert!(world.scene.joint_broken(joint).unwrap());
    assert_eq!(world.joint_break_events().len(), 1);
    assert!(world.is_sleeping(other));
    assert!(world.joint_reaction_samples().is_none());
}

#[test]
fn subthreshold_frames_retain_model_identity_and_match_the_ordinary_cpu_path() {
    let fixture = || {
        let mut world = world();
        let (body, id) = fixed(&mut world);
        world.scene.remove_joint(id).unwrap();
        let id = world
            .scene
            .insert_joint(JointDesc {
                parent: None,
                child: body,
                kind: JointType::Ball,
                axis: Vec3::Z,
                frame_parent: Transform::IDENTITY,
                frame_child: Transform::IDENTITY,
            })
            .unwrap();
        world
            .scene
            .add_body_wrench(body, SpatialVector::new(Vec3::X * 4.0, Vec3::Y * 3.0))
            .unwrap();
        (world, body, id)
    };
    let (mut baseline, a, _) = fixture();
    let (mut breaking, b, id) = fixture();
    breaking
        .scene
        .set_joint_break_limits(id, limits(Some(100.0), None))
        .unwrap();
    let model = breaking.scene.model_stamp();
    for _ in 0..10 {
        baseline.step(0.01).unwrap();
        breaking.step(0.01).unwrap();
        assert!(
            (baseline.scene.body_velocity(a).unwrap().angular
                - breaking.scene.body_velocity(b).unwrap().angular)
                .length()
                < 1e-10
        );
        assert!(
            baseline
                .scene
                .body_pose(a)
                .unwrap()
                .rotation
                .angle_between(breaking.scene.body_pose(b).unwrap().rotation)
                .abs()
                < 1e-7
        );
        assert_eq!(breaking.scene.model_stamp(), model);
        assert!(breaking.joint_break_events().is_empty());
    }
}

#[test]
fn all_supported_constrained_joint_kinds_break_under_the_same_support_load() {
    for (kind, external) in [
        (JointType::Fixed, false),
        (JointType::Revolute, false),
        (JointType::Prismatic, false),
        (JointType::Ball, false),
        (JointType::D6, false),
        (JointType::Distance, true),
        (JointType::Ball, true),
    ] {
        let mut world = world();
        let (body, old) = fixed(&mut world);
        world.scene.remove_joint(old).unwrap();
        world.scene.set_gravity(-Vec3::Y * 10.0).unwrap();
        let desc = JointDesc {
            parent: None,
            child: body,
            kind,
            axis: Vec3::Z,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        };
        let id = if kind == JointType::Ball && external {
            world.scene.insert_rigid_loop_joint(desc).unwrap()
        } else {
            world.scene.insert_joint(desc).unwrap()
        };
        if kind == JointType::D6 {
            world
                .scene
                .set_joint_locked_axes(
                    id,
                    molla_sim::joint_control::JointAxes::NONE
                        .with(molla_sim::joint_control::JointAxis::LinY),
                )
                .unwrap();
        }
        if kind == JointType::Distance {
            world
                .scene
                .set_distance_limits(
                    id,
                    molla_sim::runtime::DistanceLimits {
                        min: None,
                        max: Some(0.0),
                    },
                )
                .unwrap();
        }
        world
            .scene
            .set_joint_break_limits(id, limits(Some(19.0), None))
            .unwrap();
        world.step(0.04).unwrap();
        assert!(
            world.scene.joint_broken(id).unwrap(),
            "{kind:?} external={external}"
        );
        assert_eq!(world.joint_break_events().len(), 1);
        assert!((world.joint_break_events()[0].sample.child.linear.y - 20.0).abs() < 1e-8);
        assert!((world.scene.body_velocity(body).unwrap().linear.y + 0.35).abs() < 1e-8);
    }
}

#[test]
fn angular_stop_peak_is_sampled_before_mean_reduction() {
    let mut world = world();
    let (body, old) = fixed(&mut world);
    world.scene.remove_joint(old).unwrap();
    let id = world
        .scene
        .insert_joint(JointDesc {
            parent: None,
            child: body,
            kind: JointType::Revolute,
            axis: Vec3::Z,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    world
        .scene
        .set_joint_limits(
            id,
            molla_sim::joint_control::JointAxis::AngX,
            Some([-1.0, 0.0]),
        )
        .unwrap();
    world
        .scene
        .set_body_velocity(body, SpatialVector::new(Vec3::ZERO, Vec3::Z * 2.0))
        .unwrap();
    world
        .scene
        .set_joint_break_limits(id, limits(None, Some(400.0)))
        .unwrap();
    world.solver.substeps = 4;
    world.step(0.04).unwrap();
    assert_eq!(
        world.joint_break_events()[0].trigger,
        JointBreakTrigger::Torque
    );
    assert!((world.joint_break_events()[0].sample.child.angular.z + 800.0).abs() < 1e-9);
    assert!((world.joint_reaction(id).unwrap().child.angular.z + 200.0).abs() < 1e-9);
    assert_eq!(
        world.scene.body_velocity(body).unwrap(),
        SpatialVector::ZERO
    );
}

#[test]
fn contact_load_triggers_break_and_all_substep_contact_samples_are_retained() {
    use molla_sim::runtime::{ColliderDesc, ColliderGeometry};
    let mut world = world();
    let (body, joint) = fixed(&mut world);
    let mut sphere = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
    sphere.parent = Some(body);
    world.scene.insert_collider(sphere).unwrap();
    let mut ground = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(2.0, 0.1, 2.0),
    });
    ground.pose = Transform::from_translation(Vec3::new(0.0, -0.55, 0.0));
    world.scene.insert_collider(ground).unwrap();
    world
        .scene
        .set_joint_break_limits(joint, limits(Some(1.0), None))
        .unwrap();
    world.solver.substeps = 4;
    world.step(0.001).unwrap();
    assert!(world.scene.joint_broken(joint).unwrap());
    assert_eq!(world.joint_break_events().len(), 1);
    let samples: Vec<_> = world
        .contact_pairs()
        .into_iter()
        .flat_map(|pair| pair.points)
        .collect();
    assert_eq!(samples.len(), 4);
    assert!(samples.iter().all(|p| p.acted && p.sample_dt == 0.00025));
    assert!(world.joint_break_events()[0].sample.child.linear.y < -1.0);
    assert!(world.scene.body_velocity(body).unwrap().linear.y > 0.0);
}

#[test]
fn sequential_break_boundaries_commit_once_or_roll_back_together() {
    for fail in [false, true] {
        let mut world = world();
        let (a, ja) = fixed(&mut world);
        let (b, jb) = fixed(&mut world);
        world.solver.substeps = 4;
        world
            .scene
            .set_joint_break_limits(ja, limits(Some(1.0), None))
            .unwrap();
        world
            .scene
            .set_joint_break_limits(jb, limits(Some(1.0), None))
            .unwrap();
        let mut calls = 0;
        let result = world.step_with_forces(
            0.04,
            |_, _| false,
            |scene, _, _, forces| {
                let substep = calls;
                calls += 1;
                if fail && substep == 3 {
                    return Err(molla_core::Error::Solver("after two detachments".into()));
                }
                if substep == 0 {
                    forces[scene.body_index(a).unwrap().index()].linear.x = 2.0;
                }
                if substep == 2 {
                    forces[scene.body_index(b).unwrap().index()].linear.x = 2.0;
                }
                Ok(())
            },
        );
        if fail {
            assert!(result.is_err());
            assert_eq!(world.completed_steps(), 0);
            assert!(!world.scene.joint_broken(ja).unwrap());
            assert!(!world.scene.joint_broken(jb).unwrap());
            assert!(world.joint_break_events().is_empty());
        } else {
            result.unwrap();
            assert_eq!(world.completed_steps(), 1);
            assert_eq!(
                world
                    .joint_break_events()
                    .iter()
                    .map(|e| (e.frame, e.joint, e.sample.substep))
                    .collect::<Vec<_>>(),
                vec![(0, ja, 0), (0, jb, 2)]
            );
            world.step(0.04).unwrap();
            assert_eq!(world.completed_steps(), 2);
            world.scene.repair_joint(ja).unwrap();
            world
                .scene
                .add_body_wrench(a, SpatialVector::new(Vec3::X * 2.0, Vec3::ZERO))
                .unwrap();
            world.step(0.04).unwrap();
            assert_eq!(world.joint_break_events().last().unwrap().frame, 2);
        }
    }
}

#[test]
fn compliant_loop_break_preserves_boundary_motion_and_legacy_frame_reactions() {
    let fixture = || {
        let mut world = world();
        let (body, old) = fixed(&mut world);
        world.scene.remove_joint(old).unwrap();
        world
            .scene
            .set_body_pose(body, Transform::from_translation(Vec3::X * 0.1))
            .unwrap();
        let joint = world
            .scene
            .insert_loop_joint(
                JointDesc {
                    parent: None,
                    child: body,
                    kind: JointType::Fixed,
                    axis: Vec3::Z,
                    frame_parent: Transform::IDENTITY,
                    frame_child: Transform::IDENTITY,
                },
                molla_sim::joint_control::JointSoftness {
                    natural_frequency: 10.0,
                    damping_ratio: 1.0,
                },
            )
            .unwrap();
        (world, body, joint)
    };
    let (mut reference, a, ja) = fixture();
    let (mut automatic, b, jb) = fixture();
    reference.solver.substeps = 1;
    reference.step(0.00125).unwrap();
    let force = reference.devices.reaction(ja).unwrap().force;
    reference.scene.break_joints(&[ja]).unwrap();
    reference.solver.substeps = 3;
    reference.step(0.00375).unwrap();
    automatic
        .scene
        .set_joint_break_limits(jb, limits(Some(1.0), None))
        .unwrap();
    automatic.solver.substeps = 4;
    automatic.step(0.005).unwrap();
    assert_eq!(automatic.joint_break_events()[0].sample.substep, 0);
    assert!(
        (automatic.scene.body_pose(b).unwrap().position
            - reference.scene.body_pose(a).unwrap().position)
            .length()
            < 1e-10
    );
    assert!(
        (automatic.scene.body_velocity(b).unwrap().linear
            - reference.scene.body_velocity(a).unwrap().linear)
            .length()
            < 1e-10
    );
    assert!((automatic.devices.reaction(jb).unwrap().force - force * 0.25).length() < 1e-9);
}

#[test]
fn surviving_motor_efforts_follow_handles_across_new_root_dof_compaction() {
    use molla_sim::joint_control::{JointAxis, JointMotor};
    let mut world = world();
    let (a, ja) = fixed(&mut world);
    let (b, old) = fixed(&mut world);
    world.scene.remove_joint(old).unwrap();
    let jb = world
        .scene
        .insert_joint(JointDesc {
            parent: None,
            child: b,
            kind: JointType::Revolute,
            axis: Vec3::Z,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    world
        .scene
        .set_joint_motor(
            jb,
            JointAxis::AngX,
            JointMotor {
                mode: molla_sim::JointTargetMode::Effort,
                target_position: 100.0,
                max_force: 7.0,
                ..Default::default()
            },
        )
        .unwrap();
    world
        .scene
        .add_body_wrench(b, SpatialVector::new(Vec3::ZERO, -Vec3::Z * 5.0))
        .unwrap();
    world
        .scene
        .set_joint_break_limits(ja, limits(Some(1.0), None))
        .unwrap();
    world.solver.substeps = 4;
    let mut calls = 0;
    world
        .step_with_forces(
            0.04,
            |_, _| false,
            |scene, _, _, forces| {
                if calls == 1 {
                    forces[scene.body_index(a).unwrap().index()].linear.x = 2.0;
                }
                calls += 1;
                Ok(())
            },
        )
        .unwrap();
    let index = world.scene.joint_index(jb).unwrap().index();
    let dof = world.scene.model().joint_dof_offset.host().unwrap()[index] as usize;
    assert!(dof >= 6);
    assert_eq!(world.solver.joint_efforts()[dof].drive, 7.0);
    assert!((world.joint_reaction(jb).unwrap().child.angular.z - 7.0).abs() < 1e-10);
    assert!((world.scene.body_velocity(b).unwrap().angular.z - 0.02).abs() < 1e-10);
}

#[test]
fn automatic_break_refreshes_collision_policy_before_the_following_substep() {
    use molla_sim::runtime::{ColliderDesc, ColliderGeometry};
    for explicit in [false, true] {
        let mut world = world();
        let (a, old_a) = fixed(&mut world);
        let (b, old_b) = fixed(&mut world);
        world.scene.remove_joint(old_a).unwrap();
        world.scene.remove_joint(old_b).unwrap();
        let joint = world
            .scene
            .insert_joint(JointDesc {
                parent: Some(a),
                child: b,
                kind: JointType::Fixed,
                axis: Vec3::Z,
                frame_parent: Transform::IDENTITY,
                frame_child: Transform::IDENTITY,
            })
            .unwrap();
        for body in [a, b] {
            let mut collider = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.1 });
            collider.parent = Some(body);
            world.scene.insert_collider(collider).unwrap();
        }
        if explicit {
            world.scene.set_body_pair_excluded(a, b, true).unwrap();
        }
        world
            .scene
            .add_body_wrench(b, SpatialVector::new(Vec3::X * 8.0, Vec3::ZERO))
            .unwrap();
        world
            .scene
            .set_joint_break_limits(joint, limits(Some(3.0), None))
            .unwrap();
        world.solver.substeps = 4;
        world.step(0.0001).unwrap();
        assert_eq!(world.joint_break_events().len(), 1);
        assert_eq!(world.joint_break_events()[0].sample.substep, 0);
        assert!((world.joint_break_events()[0].sample.child.linear.x + 4.0).abs() < 1e-9);
        assert_eq!(world.contact_pairs().is_empty(), explicit);
    }
}
