use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxis, JointMotor, JointSoftness};
use molla_sim::runtime::{
    BodyDesc, BodyHandle, BodyKind, ColliderDesc, ColliderGeometry, DistanceLimits, JointDesc,
    JointHandle, SphericalConeLimits,
};
use molla_sim::{JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

fn world() -> RigidWorld {
    let mut world = RigidWorld::new().unwrap();
    world.scene.set_gravity(Vec3::ZERO).unwrap();
    world.set_reaction_readout(true).unwrap();
    world
}

fn body(world: &mut RigidWorld) -> BodyHandle {
    world
        .scene
        .insert_body(BodyDesc {
            additional_mass: Some(MassProperties {
                mass: 2.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * 4.0,
            }),
            ..Default::default()
        })
        .unwrap()
}

fn desc(parent: Option<BodyHandle>, child: BodyHandle, kind: JointType) -> JointDesc {
    JointDesc {
        parent,
        child,
        kind,
        axis: Vec3::X,
        frame_parent: Transform::IDENTITY,
        frame_child: Transform::IDENTITY,
    }
}

fn near(actual: SpatialVector, expected: SpatialVector, tolerance: f64) {
    assert!(
        (actual.linear - expected.linear).length() < tolerance,
        "{actual:?} != {expected:?}"
    );
    assert!(
        (actual.angular - expected.angular).length() < tolerance,
        "{actual:?} != {expected:?}"
    );
}

fn reaction(world: &RigidWorld, joint: JointHandle, expected: SpatialVector) {
    let reading = world.joint_reaction(joint).unwrap();
    near(reading.child, expected, 1e-8);
    near(reading.parent, expected.scale(-1.0), 1e-8);
}

#[test]
fn independent_articulation_reactions_keep_owners_after_compaction() {
    let mut world = world();
    world.solver.substeps = 4;
    let mut owners = Vec::new();
    for index in 0..16 {
        let body = body(&mut world);
        let pose = Transform::from_translation(Vec3::X * (index as f64 * 3.));
        world.scene.set_body_pose(body, pose).unwrap();
        let mut joint = desc(None, body, JointType::Fixed);
        joint.frame_parent = pose;
        let joint = world.scene.insert_joint(joint).unwrap();
        let load = SpatialVector::new(Vec3::X * (index + 1) as f64, Vec3::ZERO);
        world.scene.add_body_wrench(body, load).unwrap();
        owners.push((body, joint, load));
    }
    for remove in [None, Some(5)] {
        if let Some(index) = remove {
            let (body, _, _) = owners.remove(index);
            world.scene.remove_body(body).unwrap();
        }
        world.step(0.02).unwrap();
        assert_eq!(
            world.joint_reaction_samples().unwrap().len(),
            owners.len() * 4
        );
        for &(_, joint, load) in &owners {
            reaction(&world, joint, load.scale(-1.));
            let samples: Vec<_> = world
                .joint_reaction_samples()
                .unwrap()
                .iter()
                .filter(|sample| sample.joint == joint)
                .collect();
            assert_eq!(samples.len(), 4);
            for sample in samples {
                near(sample.child, load.scale(-1.), 1e-8);
            }
        }
    }
}

#[test]
fn fixed_support_reports_gravity_loads_and_transient_substep_peaks() {
    let mut world = world();
    let child = body(&mut world);
    let joint = world
        .scene
        .insert_joint(desc(None, child, JointType::Fixed))
        .unwrap();
    world.scene.set_gravity(-Vec3::Y * 10.0).unwrap();
    world.solver.substeps = 4;
    let mut substep = 0;
    world
        .step_with_forces(
            0.04,
            |_, _| false,
            |scene, _, _, forces| {
                forces[scene.body_index(child).unwrap().index()].linear +=
                    Vec3::X * [0.0, 8.0, 0.0, 0.0][substep];
                substep += 1;
                Ok(())
            },
        )
        .unwrap();
    reaction(
        &world,
        joint,
        SpatialVector::new(Vec3::new(-2.0, 20.0, 0.0), Vec3::ZERO),
    );
    let reading = world.joint_reaction(joint).unwrap();
    assert!((reading.peak_force - 464.0_f64.sqrt()).abs() < 1e-10);
    assert!((reading.sample_dt - 0.04).abs() < 1e-12);
    let samples = world.joint_reaction_samples().unwrap();
    assert_eq!(samples.len(), 4);
    for (index, sample) in samples.iter().enumerate() {
        assert_eq!(sample.substep, index);
        assert_eq!(sample.time_offset, (index + 1) as f64 * 0.01);
        assert_eq!(sample.sample_dt, 0.01);
    }
    assert_eq!(samples[1].child.linear.x, -8.0);
}

#[test]
fn scalar_stops_report_actual_load_with_no_commanded_drive() {
    for kind in [JointType::Prismatic, JointType::Revolute] {
        let mut world = world();
        let a = body(&mut world);
        let b = body(&mut world);
        let joint = world.scene.insert_joint(desc(Some(a), b, kind)).unwrap();
        let (axis, wrench) = if kind == JointType::Prismatic {
            (
                JointAxis::LinX,
                SpatialVector::new(Vec3::X * 8.0, Vec3::ZERO),
            )
        } else {
            (
                JointAxis::AngX,
                SpatialVector::new(Vec3::ZERO, Vec3::X * 8.0),
            )
        };
        world
            .scene
            .set_joint_limits(joint, axis, Some([-1.0, 0.0]))
            .unwrap();
        world.scene.add_body_wrench(b, wrench).unwrap();
        world.step(0.01).unwrap();
        reaction(&world, joint, wrench.scale(-0.5));
        assert!(
            world
                .scene
                .control()
                .joint_target_mode
                .host()
                .unwrap()
                .iter()
                .all(|&mode| mode == 0)
        );
    }
}

#[test]
fn ball_and_free_d6_report_only_transmitted_components() {
    for kind in [JointType::Ball, JointType::D6] {
        let mut world = world();
        let child = body(&mut world);
        let joint = world.scene.insert_joint(desc(None, child, kind)).unwrap();
        let force = Vec3::new(3.0, -4.0, 5.0);
        world
            .scene
            .add_body_wrench(child, SpatialVector::new(force, Vec3::X * 6.0))
            .unwrap();
        world.step(0.01).unwrap();
        reaction(
            &world,
            joint,
            SpatialVector::new(
                if kind == JointType::Ball {
                    -force
                } else {
                    Vec3::ZERO
                },
                Vec3::ZERO,
            ),
        );
    }
}

#[test]
fn reversed_joint_preserves_authored_child_sign_and_anchor_moments() {
    let mut world = world();
    let a = body(&mut world);
    let b = body(&mut world);
    let mut joint_desc = desc(Some(a), b, JointType::Fixed);
    joint_desc.frame_parent.position = Vec3::Y;
    joint_desc.frame_child.position = Vec3::Y;
    let joint = world.scene.insert_joint(joint_desc).unwrap();
    world.scene.set_body_kind(b, BodyKind::Fixed).unwrap();
    let index = world.scene.joint_index(joint).unwrap().index();
    assert_ne!(world.scene.model().joint_reversed.host().unwrap()[index], 0);
    world
        .scene
        .add_body_wrench(a, SpatialVector::new(Vec3::X * 6.0, Vec3::Z * 2.0))
        .unwrap();
    world.step(0.01).unwrap();
    reaction(
        &world,
        joint,
        SpatialVector::new(Vec3::X * 6.0, Vec3::Z * 8.0),
    );
}

#[test]
fn external_ball_and_compliant_loop_loads_are_not_double_counted() {
    for compliant in [false, true] {
        let mut world = world();
        let a = body(&mut world);
        let b = body(&mut world);
        let tree = world
            .scene
            .insert_joint(desc(Some(a), b, JointType::Fixed))
            .unwrap();
        let closure = desc(None, b, JointType::Ball);
        let link = if compliant {
            world
                .scene
                .insert_loop_joint(
                    closure,
                    JointSoftness {
                        natural_frequency: 30.0,
                        damping_ratio: 1.0,
                    },
                )
                .unwrap()
        } else {
            world.scene.insert_rigid_loop_joint(closure).unwrap()
        };
        world
            .scene
            .add_body_wrench(b, SpatialVector::new(Vec3::Y * 8.0, Vec3::ZERO))
            .unwrap();
        world.step(0.01).unwrap();
        let load = world.joint_reaction(link).unwrap().child.linear.y;
        let tree_load = world.joint_reaction(tree).unwrap().child.linear.y;
        assert!(
            (tree_load + (8.0 + load) * 0.5).abs() < 1e-8,
            "compliant={compliant} loop={load} tree={tree_load}"
        );
        if !compliant {
            assert!((load + 8.0).abs() < 1e-8);
        }
    }
}

#[test]
fn stale_failed_disabled_and_sleeping_frames_do_not_leak_samples() {
    let mut world = world();
    let child = body(&mut world);
    let joint = world
        .scene
        .insert_joint(desc(None, child, JointType::Ball))
        .unwrap();
    assert!(world.joint_reaction_samples().is_none());
    world.step(0.01).unwrap();
    assert!(world.joint_reaction(joint).is_some());
    world.scene.set_gravity(-Vec3::Y).unwrap();
    assert!(world.joint_reaction_samples().is_none());
    world.step(0.01).unwrap();
    assert!(world.step(f64::NAN).is_err());
    assert!(world.joint_reaction_samples().is_none());
    world.step(0.01).unwrap();
    assert!(
        world
            .step_with_forces(
                0.01,
                |_, _| false,
                |_, _, _, _| Err(molla_core::Error::Solver("test fault".into()))
            )
            .is_err()
    );
    assert!(world.joint_reaction_samples().is_none());
    world.sleep_body(child).unwrap();
    world.step(0.01).unwrap();
    assert!(world.joint_reaction(joint).is_none());
    assert!(world.joint_reaction_samples().unwrap().is_empty());
    world.wake_body(child).unwrap();
    world.step(0.01).unwrap();
    assert!(world.joint_reaction(joint).is_some());
    world.set_reaction_readout(false).unwrap();
    world.step(0.01).unwrap();
    assert!(world.joint_reaction_samples().is_none());
}

#[test]
fn kinematic_only_scene_preserves_empty_reaction_readout() {
    let mut world = world();
    let child = body(&mut world);
    world
        .scene
        .set_body_kind(child, BodyKind::Kinematic)
        .unwrap();
    let mut called = false;
    assert!(
        world
            .step_with_forces(
                0.01,
                |_, _| false,
                |_, _, _, _| {
                    called = true;
                    Ok(())
                }
            )
            .is_ok()
    );
    assert!(called);
    assert!(world.joint_reaction_samples().unwrap().is_empty());
    world.set_reaction_readout(false).unwrap();
    world.set_reaction_readout(true).unwrap();
    assert!(world.reaction_readout_enabled());
}

#[test]
fn input_edits_and_dense_compaction_invalidate_reactions_without_requiring_wake() {
    let mut world = world();
    let removed = body(&mut world);
    let child = body(&mut world);
    let joint = world
        .scene
        .insert_joint(desc(None, child, JointType::Ball))
        .unwrap();
    world.step(0.01).unwrap();
    world
        .scene
        .add_body_wrench_with_wake(child, SpatialVector::new(Vec3::X, Vec3::ZERO), false)
        .unwrap();
    assert!(world.joint_reaction(joint).is_none());
    world.step(0.01).unwrap();
    world
        .scene
        .reset_body_forces_with_wake(child, false)
        .unwrap();
    assert!(world.joint_reaction(joint).is_none());
    world.step(0.01).unwrap();
    world
        .scene
        .set_joint_motor_with_wake(
            joint,
            JointAxis::AngX,
            JointMotor {
                mode: molla_sim::JointTargetMode::Effort,
                target_position: 3.0,
                ..Default::default()
            },
            false,
        )
        .unwrap();
    assert!(world.joint_reaction(joint).is_none());
    world.step(0.01).unwrap();
    world.scene.control_mut();
    assert!(world.joint_reaction(joint).is_none());
    world.step(0.01).unwrap();
    world.scene.state_mut();
    assert!(world.joint_reaction(joint).is_none());
    world.step(0.01).unwrap();
    world.scene.remove_body(removed).unwrap();
    assert!(world.joint_reaction(joint).is_none());
    world.step(0.01).unwrap();
    assert!(world.joint_reaction(joint).is_some());
    world.scene.remove_joint(joint).unwrap();
    assert!(world.joint_reaction(joint).is_none());
}

#[test]
fn cone_only_soft_loop_and_rigid_loop_report_the_same_angular_load() {
    for compliant in [false, true] {
        let mut world = world();
        let child = body(&mut world);
        let closure = desc(None, child, JointType::Ball);
        let joint = if compliant {
            world
                .scene
                .insert_loop_joint(
                    closure,
                    JointSoftness {
                        natural_frequency: 0.0,
                        damping_ratio: 0.0,
                    },
                )
                .unwrap()
        } else {
            world.scene.insert_rigid_loop_joint(closure).unwrap()
        };
        world
            .scene
            .set_cone_limits(
                joint,
                SphericalConeLimits {
                    angle0: Some(0.0),
                    angle1: Some(0.0),
                },
            )
            .unwrap();
        world
            .scene
            .add_body_wrench(child, SpatialVector::new(Vec3::ZERO, Vec3::Y * 4.0))
            .unwrap();
        world.step(0.01).unwrap();
        reaction(
            &world,
            joint,
            SpatialVector::new(Vec3::ZERO, -Vec3::Y * 4.0),
        );
    }
}

#[test]
fn radial_loop_samples_sleep_without_affecting_an_awake_group() {
    let mut world = world();
    let child = body(&mut world);
    world
        .scene
        .set_body_pose(child, Transform::from_translation(Vec3::X))
        .unwrap();
    let joint = world
        .scene
        .insert_joint(desc(None, child, JointType::Distance))
        .unwrap();
    world
        .scene
        .set_distance_limits(
            joint,
            DistanceLimits {
                min: Some(1.0),
                max: Some(1.0),
            },
        )
        .unwrap();
    world
        .scene
        .add_body_wrench(child, SpatialVector::new(Vec3::X * 3.0, Vec3::ZERO))
        .unwrap();
    let other = body(&mut world);
    let awake = world
        .scene
        .insert_joint(desc(None, other, JointType::Ball))
        .unwrap();
    world.step(0.01).unwrap();
    reaction(
        &world,
        joint,
        SpatialVector::new(-Vec3::X * 3.0, Vec3::ZERO),
    );
    world.sleep_body(child).unwrap();
    world.step(0.01).unwrap();
    assert!(world.joint_reaction(joint).is_none());
    assert!(world.joint_reaction(awake).is_some());
}

#[test]
fn contact_load_is_subtracted_from_support_reconstruction() {
    let mut world = world();
    let child = body(&mut world);
    let joint = world
        .scene
        .insert_joint(desc(None, child, JointType::Fixed))
        .unwrap();
    let mut sphere = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
    sphere.parent = Some(child);
    world.scene.insert_collider(sphere).unwrap();
    let mut ground = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(2.0, 0.5, 2.0),
    });
    ground.pose.position = -Vec3::Y * 0.9;
    world.scene.insert_collider(ground).unwrap();
    world.step(0.001).unwrap();
    let mut contact = SpatialVector::ZERO;
    let mut count = 0;
    for pair in world.contact_pairs() {
        for point in pair.points.iter().filter(|point| point.acted) {
            let sign = if pair.bodies[1] == Some(child) {
                1.0
            } else {
                -1.0
            };
            let force = point.force_on_b * sign * point.sample_dt / 0.001;
            contact = contact + SpatialVector::new(force, point.point.cross(force));
            count += 1;
        }
    }
    assert!(count > 0);
    assert!(contact.linear.length() > 1.0);
    reaction(&world, joint, contact.scale(-1.0));
}
