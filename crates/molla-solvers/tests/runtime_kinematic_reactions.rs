use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, BodyHandle, BodyKind, JointDesc, JointHandle, RigidScene};
use molla_sim::{BodyParams, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

fn fixture(
    kind: JointType,
    shift: Vec3,
    rotation: Quat,
) -> (RigidScene, BodyHandle, BodyHandle, JointHandle) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let com = Vec3::Y * 0.25;
    let parent = scene
        .insert_body(BodyDesc {
            kind: BodyKind::Kinematic,
            params: BodyParams {
                pose: Transform {
                    position: shift,
                    rotation,
                },
                ..Default::default()
            },
            additional_mass: Some(MassProperties {
                mass: 3.0,
                local_com: com,
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap();
    let child = scene
        .insert_body(BodyDesc {
            params: BodyParams {
                pose: Transform {
                    position: shift + rotation * com,
                    rotation,
                },
                ..Default::default()
            },
            additional_mass: Some(MassProperties {
                mass: 2.0,
                local_com: Vec3::X,
                inertia: Mat3::IDENTITY * 4.0,
            }),
            ..Default::default()
        })
        .unwrap();
    let joint = scene
        .insert_joint(JointDesc {
            parent: Some(parent),
            child,
            kind,
            axis: Vec3::Z,
            frame_parent: Transform::from_translation(com),
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    scene
        .set_body_velocity(
            parent,
            SpatialVector::new(rotation * Vec3::X * 3.0, rotation * Vec3::Z * 2.0),
        )
        .unwrap();
    (scene, parent, child, joint)
}

fn near(actual: Vec3, expected: Vec3, tolerance: f64) {
    assert!(
        (actual - expected).length() < tolerance,
        "actual={actual:?} expected={expected:?}"
    );
}

#[test]
fn driven_com_motion_advances_once_per_substep_and_redrift() {
    for substeps in [1, 4, 8] {
        let rotation = Quat::from_rotation_y(0.5);
        let (mut scene, parent, _, _) =
            fixture(JointType::Fixed, Vec3::new(32., -16., 8.), rotation);
        let acceleration = SpatialVector::new(Vec3::new(3., -2., 1.), rotation * Vec3::Z * 0.7);
        scene
            .set_kinematic_acceleration(parent, Some(acceleration))
            .unwrap();
        let pose = scene.body_pose(parent).unwrap();
        let center = scene.body_world_com(parent).unwrap();
        let velocity = scene.body_velocity(parent).unwrap();
        let mut world = RigidWorld::from_scene(scene).unwrap();
        world.solver.substeps = substeps;
        let dt = 0.08;
        world.step(dt).unwrap();
        near(
            world.scene.body_world_com(parent).unwrap(),
            center + velocity.linear * dt + acceleration.linear * (0.5 * dt * dt),
            1e-10,
        );
        near(
            world.scene.body_velocity(parent).unwrap().linear,
            velocity.linear + acceleration.linear * dt,
            1e-10,
        );
        near(
            world.scene.body_velocity(parent).unwrap().angular,
            velocity.angular + acceleration.angular * dt,
            1e-10,
        );
        let expected =
            Quat::from_scaled_axis((velocity.angular + acceleration.angular * (0.5 * dt)) * dt)
                * pose.rotation;
        assert!(
            world
                .scene
                .body_pose(parent)
                .unwrap()
                .rotation
                .dot(expected)
                .abs()
                > 1. - 1e-12
        );
        let input = world.scene.state().clone_host().unwrap();
        let mut output = input.clone_host().unwrap();
        world
            .solver
            .drift_cpu(world.scene.model(), &input, dt, &mut output)
            .unwrap();
        let once = output.body_q.host().unwrap().to_vec();
        world
            .solver
            .drift_cpu(world.scene.model(), &input, dt, &mut output)
            .unwrap();
        assert_eq!(once, output.body_q.host().unwrap());
    }
}

#[test]
fn driven_fixed_hinge_ball_reactions_include_boundary_acceleration() {
    for kind in [JointType::Fixed, JointType::Revolute, JointType::Ball] {
        for shift in [Vec3::ZERO, Vec3::new(32., -16., 8.)] {
            let (mut scene, parent, _, joint) = fixture(kind, shift, Quat::IDENTITY);
            scene
                .set_body_velocity(parent, SpatialVector::ZERO)
                .unwrap();
            scene
                .set_kinematic_acceleration(
                    parent,
                    Some(SpatialVector::new(Vec3::new(3., 6., 0.), Vec3::Z * 2.)),
                )
                .unwrap();
            let mut world = RigidWorld::from_scene(scene).unwrap();
            world.solver.substeps = 1;
            world.set_reaction_readout(true).unwrap();
            world.step(0.001).unwrap();
            let reading = world.joint_reaction(joint).unwrap();
            let fixed = kind == JointType::Fixed;
            near(
                reading.child.linear,
                Vec3::new(6., if fixed { 16. } else { 8. }, 0.),
                1e-8,
            );
            near(
                reading.child.angular,
                Vec3::Z * if fixed { 24. } else { 0. },
                1e-8,
            );
            if !fixed {
                let index = world.scene.joint_index(joint).unwrap().index();
                let start = world.scene.model().joint_dof_offset.host().unwrap()[index] as usize;
                let axis = if kind == JointType::Ball { 2 } else { 0 };
                assert!(
                    (world.scene.state().joint_qd.host().unwrap()[start + axis] + 0.004).abs()
                        < 1e-10
                );
            }
        }
    }
}

#[test]
fn driven_motion_edits_are_atomic_and_survive_fork_and_disable() {
    let (mut scene, parent, child, _) = fixture(JointType::Fixed, Vec3::ZERO, Quat::IDENTITY);
    let acceleration = SpatialVector::new(Vec3::X, Vec3::ZERO);
    scene
        .set_kinematic_acceleration(parent, Some(acceleration))
        .unwrap();
    let stamp = scene.dynamics_stamp();
    assert!(
        scene
            .set_kinematic_acceleration(child, Some(acceleration))
            .is_err()
    );
    assert!(
        scene
            .set_kinematic_acceleration(
                parent,
                Some(SpatialVector::new(Vec3::splat(f64::NAN), Vec3::ZERO))
            )
            .is_err()
    );
    assert_eq!(stamp, scene.dynamics_stamp());
    let mut fork = scene.fork().unwrap();
    assert_eq!(
        fork.kinematic_acceleration(parent).unwrap(),
        Some(acceleration)
    );
    fork.set_body_enabled(parent, false).unwrap();
    let pose = fork.body_pose(parent).unwrap();
    let mut world = RigidWorld::from_scene(fork).unwrap();
    world.step(0.01).unwrap();
    assert_eq!(world.scene.body_pose(parent).unwrap(), pose);
    assert_eq!(
        world.scene.kinematic_acceleration(parent).unwrap(),
        Some(acceleration)
    );
    world.scene.set_body_enabled(parent, true).unwrap();
    world.step(0.01).unwrap();
    assert_ne!(world.scene.body_pose(parent).unwrap(), pose);
    world.scene.set_body_kind(parent, BodyKind::Fixed).unwrap();
    assert_eq!(world.scene.kinematic_acceleration(parent).unwrap(), None);
    world.scene.remove_body(parent).unwrap();
    assert!(world.scene.kinematic_acceleration(parent).is_err());
    assert_eq!(
        scene.kinematic_acceleration(parent).unwrap(),
        Some(acceleration)
    );
}

#[test]
fn driven_break_continues_parent_motion_and_preserves_detached_velocity() {
    for substeps in [1, 4, 8] {
        let (mut scene, parent, child, joint) =
            fixture(JointType::Fixed, Vec3::ZERO, Quat::IDENTITY);
        scene
            .set_body_velocity(parent, SpatialVector::ZERO)
            .unwrap();
        scene
            .set_kinematic_acceleration(parent, Some(SpatialVector::new(Vec3::X * 3., Vec3::ZERO)))
            .unwrap();
        scene
            .set_joint_break_limits(
                joint,
                molla_sim::runtime::JointBreakLimits {
                    force: Some(5.9),
                    torque: None,
                },
            )
            .unwrap();
        let parent_center = scene.body_world_com(parent).unwrap();
        let child_center = scene.body_world_com(child).unwrap();
        let mut world = RigidWorld::from_scene(scene).unwrap();
        world.solver.substeps = substeps;
        let dt = 0.08;
        let h = dt / substeps as f64;
        world.step(dt).unwrap();
        assert_eq!(world.joint_break_events().len(), 1);
        near(
            world.joint_break_events()[0].sample.child.linear,
            Vec3::X * 6.,
            1e-10,
        );
        near(
            world.scene.body_world_com(parent).unwrap(),
            parent_center + Vec3::X * (1.5 * dt * dt),
            1e-10,
        );
        near(
            world.scene.body_velocity(child).unwrap().linear,
            Vec3::X * (3. * h),
            1e-10,
        );
        near(
            world.scene.body_world_com(child).unwrap(),
            child_center + Vec3::X * (1.5 * h * h + 3. * h * (dt - h)),
            1e-10,
        );
        world.step(dt).unwrap();
        assert_eq!(world.joint_break_events().len(), 1);
    }
}

#[test]
fn driven_acceleration_wakes_initially_stationary_dependents() {
    let (mut scene, parent, child, _) = fixture(JointType::Fixed, Vec3::ZERO, Quat::IDENTITY);
    scene
        .set_body_velocity(parent, SpatialVector::ZERO)
        .unwrap();
    scene
        .set_kinematic_acceleration(parent, Some(SpatialVector::new(Vec3::X, Vec3::ZERO)))
        .unwrap();
    let mut world = RigidWorld::from_scene(scene).unwrap();
    world.sleep_body(child).unwrap();
    world.step(0.01).unwrap();
    assert!(!world.is_sleeping(child));
    near(
        world.scene.body_velocity(child).unwrap().linear,
        Vec3::X * 0.01,
        1e-10,
    );
}

#[test]
fn driven_isolated_body_and_held_mode_have_distinct_motion() {
    let mut scene = RigidScene::new().unwrap();
    let body = scene
        .insert_body(BodyDesc {
            kind: BodyKind::Kinematic,
            ..Default::default()
        })
        .unwrap();
    scene
        .set_body_velocity(body, SpatialVector::new(Vec3::X, Vec3::ZERO))
        .unwrap();
    let mut world = RigidWorld::from_scene(scene).unwrap();
    world.step(0.1).unwrap();
    near(
        world.scene.body_pose(body).unwrap().position,
        Vec3::ZERO,
        1e-12,
    );
    world
        .scene
        .set_kinematic_acceleration(body, Some(SpatialVector::ZERO))
        .unwrap();
    world.step(0.1).unwrap();
    near(
        world.scene.body_pose(body).unwrap().position,
        Vec3::X * 0.1,
        1e-12,
    );
    world.scene.set_kinematic_acceleration(body, None).unwrap();
    world.step(0.1).unwrap();
    near(
        world.scene.body_pose(body).unwrap().position,
        Vec3::X * 0.1,
        1e-12,
    );
}

#[test]
fn driven_post_break_failure_restores_motion_and_configuration() {
    let (mut scene, parent, child, joint) = fixture(JointType::Fixed, Vec3::ZERO, Quat::IDENTITY);
    let acceleration = SpatialVector::new(Vec3::X * 3., Vec3::ZERO);
    scene
        .set_kinematic_acceleration(parent, Some(acceleration))
        .unwrap();
    scene
        .set_joint_break_limits(
            joint,
            molla_sim::runtime::JointBreakLimits {
                force: Some(1.),
                torque: None,
            },
        )
        .unwrap();
    let mut world = RigidWorld::from_scene(scene).unwrap();
    world.solver.substeps = 4;
    let poses = [parent, child].map(|b| world.scene.body_pose(b).unwrap());
    let velocities = [parent, child].map(|b| world.scene.body_velocity(b).unwrap());
    let stamp = world.scene.dynamics_stamp();
    let mut calls = 0;
    let result = world.step_with_forces(
        0.04,
        |_, _| false,
        |scene, _, _, _| {
            calls += 1;
            if calls == 3 {
                assert!(scene.joint_broken(joint).unwrap());
                return Err(molla_core::Error::Solver(
                    "injected driven post-break failure".into(),
                ));
            }
            Ok(())
        },
    );
    assert!(result.is_err());
    assert_eq!(calls, 3);
    assert_eq!(world.scene.dynamics_stamp(), stamp);
    assert_eq!(
        [parent, child].map(|b| world.scene.body_pose(b).unwrap()),
        poses
    );
    assert_eq!(
        [parent, child].map(|b| world.scene.body_velocity(b).unwrap()),
        velocities
    );
    assert_eq!(
        world.scene.kinematic_acceleration(parent).unwrap(),
        Some(acceleration)
    );
    assert!(!world.scene.joint_broken(joint).unwrap());
    assert!(world.joint_break_events().is_empty());
    world.step(0.04).unwrap();
    assert!(world.scene.joint_broken(joint).unwrap());
}

#[test]
fn noncollinear_angular_acceleration_has_second_order_orientation_convergence() {
    let mut scene = RigidScene::new().unwrap();
    let body = scene
        .insert_body(BodyDesc {
            kind: BodyKind::Kinematic,
            ..Default::default()
        })
        .unwrap();
    scene
        .set_body_velocity(
            body,
            SpatialVector::new(Vec3::ZERO, Vec3::new(0.7, -0.4, 1.2)),
        )
        .unwrap();
    scene
        .set_kinematic_acceleration(
            body,
            Some(SpatialVector::new(Vec3::ZERO, Vec3::new(-0.5, 0.8, 0.3))),
        )
        .unwrap();
    let index = scene.body_index(body).unwrap().index();
    let integrate = |steps| {
        let solver = molla_solvers::FeatherstoneSolver::new_cpu();
        let mut input = scene.state().clone_host().unwrap();
        let mut output = input.clone_host().unwrap();
        for _ in 0..steps {
            solver
                .drift_cpu(scene.model(), &input, 1. / steps as f64, &mut output)
                .unwrap();
            std::mem::swap(&mut input, &mut output);
        }
        input.body_q.host().unwrap()[index].rotation
    };
    let reference = integrate(16384);
    let errors = [16, 32, 64].map(|steps| {
        let delta = integrate(steps) * reference.conjugate();
        2. * Vec3::new(delta.x, delta.y, delta.z)
            .length()
            .atan2(delta.w.abs())
    });
    assert!(
        errors[0] / errors[1] > 3.9 && errors[1] / errors[2] > 3.9,
        "errors={errors:?}"
    );
    eprintln!("prescribed noncollinear orientation errors={errors:?}");
}

#[test]
fn translating_rotating_kinematic_root_does_not_add_spurious_hinge_acceleration() {
    for shift in [Vec3::ZERO, Vec3::new(32.0, -16.0, 8.0)] {
        for rotation in [
            Quat::IDENTITY,
            Quat::from_rotation_y(0.5) * Quat::from_rotation_x(-0.4),
        ] {
            let (scene, parent, child, joint) = fixture(JointType::Revolute, shift, rotation);
            let mut world = RigidWorld::from_scene(scene).unwrap();
            world.solver.substeps = 1;
            let pose = world.scene.body_pose(parent).unwrap();
            let velocity = world.scene.body_velocity(parent).unwrap();
            world.step(0.001).unwrap();
            let index = world.scene.joint_index(joint).unwrap().index();
            let offset = world.scene.model().joint_dof_offset.host().unwrap()[index] as usize;
            let rate = world.scene.state().joint_qd.host().unwrap()[offset];
            assert!(
                rate.abs() < 1e-10,
                "shift={shift:?} rotation={rotation:?} spurious hinge rate={rate}"
            );
            near(
                world.scene.body_velocity(child).unwrap().angular,
                rotation * Vec3::Z * 2.0,
                1e-10,
            );
            assert_eq!(world.scene.body_pose(parent).unwrap(), pose);
            assert_eq!(world.scene.body_velocity(parent).unwrap(), velocity);
        }
    }
}

#[test]
fn moving_kinematic_tree_reactions_match_centripetal_support_loads() {
    for kind in [JointType::Fixed, JointType::Revolute, JointType::Ball] {
        for substeps in [1, 4, 8] {
            for shift in [Vec3::ZERO, Vec3::new(32.0, -16.0, 8.0)] {
                let rotation = Quat::from_rotation_y(0.5) * Quat::from_rotation_x(-0.4);
                let (scene, parent, _, joint) = fixture(kind, shift, rotation);
                let mut world = RigidWorld::from_scene(scene).unwrap();
                world.solver.substeps = substeps;
                world.set_reaction_readout(true).unwrap();
                let prescribed = world.scene.body_velocity(parent).unwrap();
                world.step(0.001).unwrap();
                let reading = world.joint_reaction(joint).unwrap();
                near(reading.child.linear, rotation * -Vec3::X * 8.0, 1e-8);
                near(reading.child.angular, Vec3::ZERO, 1e-8);
                near(reading.parent.linear, -reading.child.linear, 1e-10);
                near(reading.parent.angular, Vec3::ZERO, 1e-8);
                assert_eq!(world.joint_reaction_samples().unwrap().len(), substeps);
                assert_eq!(world.scene.body_velocity(parent).unwrap(), prescribed);
            }
        }
    }
}

#[test]
fn moving_kinematic_fixed_joint_reports_full_inertia_gyro_and_external_loads() {
    let rotation = Quat::from_rotation_y(0.5) * Quat::from_rotation_x(-0.4);
    let (mut scene, parent, child, joint) =
        fixture(JointType::Fixed, Vec3::new(32.0, -16.0, 8.0), rotation);
    let inertia = Mat3::from_cols(
        Vec3::new(2.0, 0.1, 0.2),
        Vec3::new(0.1, 3.0, -0.3),
        Vec3::new(0.2, -0.3, 4.0),
    );
    scene
        .set_body_mass_properties(
            child,
            MassProperties {
                mass: 2.0,
                local_com: Vec3::X,
                inertia,
            },
        )
        .unwrap();
    let omega = rotation * Vec3::new(0.4, -0.5, 1.2);
    scene
        .set_body_velocity(parent, SpatialVector::new(Vec3::new(1.0, -2.0, 3.0), omega))
        .unwrap();
    let gravity = -Vec3::Y * 9.81;
    scene.set_gravity(gravity).unwrap();
    let applied = SpatialVector::new(Vec3::new(2.0, -3.0, 1.0), Vec3::new(0.4, 0.1, -0.7));
    scene.add_body_wrench(child, applied).unwrap();
    let r = scene.body_world_com(child).unwrap() - scene.body_world_com(parent).unwrap();
    let rmat = Mat3::from_quat(scene.body_pose(child).unwrap().rotation);
    let iw = rmat * inertia * rmat.transpose();
    let force = (omega.cross(omega.cross(r)) - gravity) * 2.0 - applied.linear;
    let torque = omega.cross(iw * omega) - applied.angular + r.cross(force);
    let mut world = RigidWorld::from_scene(scene).unwrap();
    world.set_reaction_readout(true).unwrap();
    world.step(0.01).unwrap();
    let reading = world.joint_reaction(joint).unwrap();
    near(reading.child.linear, force, 1e-9);
    near(reading.child.angular, torque, 1e-9);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_driven_motion_reactions_break_continuation_and_disable_match_cpu() {
    let engine =
        std::sync::Arc::new(molla_compute::Engine::new(molla_compute::Backend::AutoGpu).unwrap());
    let (mut scene, parent, child, joint) = fixture(JointType::Fixed, Vec3::ZERO, Quat::IDENTITY);
    scene
        .set_body_velocity(parent, SpatialVector::ZERO)
        .unwrap();
    scene
        .set_kinematic_acceleration(parent, Some(SpatialVector::new(Vec3::X * 3., Vec3::ZERO)))
        .unwrap();
    let mut cpu = RigidWorld::from_scene(scene.fork().unwrap()).unwrap();
    let mut world = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    for w in [&mut cpu, &mut world] {
        w.solver.substeps = 4;
        w.set_reaction_readout(true).unwrap();
        w.scene
            .set_joint_break_limits(
                joint,
                molla_sim::runtime::JointBreakLimits {
                    force: Some(6.1),
                    torque: None,
                },
            )
            .unwrap();
        w.step(0.04).unwrap();
        assert!(!w.scene.joint_broken(joint).unwrap());
        near(
            w.joint_reaction(joint).unwrap().child.linear,
            Vec3::X * 6.,
            1e-5,
        );
        w.scene
            .set_joint_break_limits(
                joint,
                molla_sim::runtime::JointBreakLimits {
                    force: Some(5.9),
                    torque: None,
                },
            )
            .unwrap();
        w.step(0.04).unwrap();
        assert!(w.scene.joint_broken(joint).unwrap());
        assert_eq!(w.joint_break_events().len(), 1);
    }
    for body in [parent, child] {
        near(
            world.scene.body_world_com(body).unwrap(),
            cpu.scene.body_world_com(body).unwrap(),
            1e-6,
        );
        near(
            world.scene.body_velocity(body).unwrap().linear,
            cpu.scene.body_velocity(body).unwrap().linear,
            1e-6,
        );
    }
    assert_eq!(
        world.joint_break_events()[0].sample.substep,
        cpu.joint_break_events()[0].sample.substep
    );
    let pose = world.scene.body_pose(parent).unwrap();
    world
        .scene
        .set_kinematic_acceleration(parent, None)
        .unwrap();
    world.step(0.001).unwrap();
    assert_eq!(pose, world.scene.body_pose(parent).unwrap());
    world
        .scene
        .set_kinematic_acceleration(
            parent,
            Some(SpatialVector::new(Vec3::X * 1e100, Vec3::ZERO)),
        )
        .unwrap();
    let stamp = world.scene.dynamics_stamp();
    assert!(
        world
            .step(0.001)
            .unwrap_err()
            .to_string()
            .contains("not representable")
    );
    assert_eq!(pose, world.scene.body_pose(parent).unwrap());
    assert_eq!(stamp, world.scene.dynamics_stamp());
    world.scene.remove_body(child).unwrap();
    world
        .scene
        .set_kinematic_acceleration(parent, Some(SpatialVector::ZERO))
        .unwrap();
    let center = world.scene.body_world_com(parent).unwrap();
    let velocity = world.scene.body_velocity(parent).unwrap();
    world.step(0.01).unwrap();
    near(
        world.scene.body_world_com(parent).unwrap(),
        center + velocity.linear * 0.01,
        1e-6,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_driven_tree_angular_acceleration_matches_analytic_transmitted_loads() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let rotation = Quat::from_rotation_y(0.5) * Quat::from_rotation_x(-0.4);
    for kind in [JointType::Fixed, JointType::Revolute, JointType::Ball] {
        let (mut scene, parent, _, joint) = fixture(kind, Vec3::new(8., -4., 2.), rotation);
        scene
            .set_body_velocity(parent, SpatialVector::ZERO)
            .unwrap();
        scene
            .set_kinematic_acceleration(
                parent,
                Some(SpatialVector::new(
                    rotation * Vec3::new(3., 6., 0.),
                    rotation * Vec3::Z * 2.,
                )),
            )
            .unwrap();
        let mut world = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
        world.solver.substeps = 1;
        world.set_reaction_readout(true).unwrap();
        world.step(0.001).unwrap();
        let fixed = kind == JointType::Fixed;
        let reading = world.joint_reaction(joint).unwrap();
        let force = rotation * Vec3::new(6., if fixed { 16. } else { 8. }, 0.);
        let torque = rotation * Vec3::Z * if fixed { 24. } else { 0. };
        near(reading.child.linear, force, 3e-4 * (1. + force.length()));
        near(reading.child.angular, torque, 3e-4 * (1. + torque.length()));
        eprintln!(
            "GPU prescribed {kind:?} force_error={:e} torque_error={:e}",
            (reading.child.linear - force).length(),
            (reading.child.angular - torque).length()
        );
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_translating_rotating_kinematic_root_matches_zero_hinge_acceleration() {
    let engine =
        std::sync::Arc::new(molla_compute::Engine::new(molla_compute::Backend::AutoGpu).unwrap());
    for shift in [Vec3::ZERO, Vec3::new(32.0, -16.0, 8.0)] {
        let rotation = Quat::from_rotation_y(0.5) * Quat::from_rotation_x(-0.4);
        let (scene, parent, child, joint) = fixture(JointType::Revolute, shift, rotation);
        let mut world = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
        world.solver.substeps = 1;
        let pose = world.scene.body_pose(parent).unwrap();
        for _ in 0..20 {
            world.step(0.001).unwrap();
            let index = world.scene.joint_index(joint).unwrap().index();
            let offset = world.scene.model().joint_dof_offset.host().unwrap()[index] as usize;
            let rate = world.scene.state().joint_qd.host().unwrap()[offset];
            assert!(rate.abs() < 2e-5, "shift={shift:?} rate={rate}");
            near(
                world.scene.body_velocity(child).unwrap().angular,
                rotation * Vec3::Z * 2.0,
                2e-5,
            );
            near(
                world.scene.body_pose(parent).unwrap().position,
                pose.position,
                1e-6,
            );
        }
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_moving_kinematic_tree_reactions_match_centripetal_support_loads() {
    let engine =
        std::sync::Arc::new(molla_compute::Engine::new(molla_compute::Backend::AutoGpu).unwrap());
    for kind in [JointType::Fixed, JointType::Revolute, JointType::Ball] {
        for shift in [Vec3::ZERO, Vec3::new(32.0, -16.0, 8.0)] {
            let rotation = Quat::from_rotation_y(0.5) * Quat::from_rotation_x(-0.4);
            let (scene, parent, _, joint) = fixture(kind, shift, rotation);
            let mut world = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
            world.set_reaction_readout(true).unwrap();
            let pose = world.scene.body_pose(parent).unwrap();
            let prescribed = world.scene.body_velocity(parent).unwrap();
            for substeps in [1, 4, 8] {
                world.solver.substeps = substeps;
                world.step(0.001).unwrap();
                let reading = world.joint_reaction(joint).unwrap();
                near(reading.child.linear, rotation * -Vec3::X * 8.0, 5e-4);
                near(reading.child.angular, Vec3::ZERO, 5e-4);
                near(reading.parent.linear, -reading.child.linear, 1e-7);
                near(reading.parent.angular, Vec3::ZERO, 5e-4);
                assert_eq!(world.joint_reaction_samples().unwrap().len(), substeps);
                near(
                    world.scene.body_pose(parent).unwrap().position,
                    pose.position,
                    1e-6,
                );
                near(
                    world.scene.body_velocity(parent).unwrap().linear,
                    prescribed.linear,
                    1e-6,
                );
                near(
                    world.scene.body_velocity(parent).unwrap().angular,
                    prescribed.angular,
                    1e-6,
                );
            }
        }
    }
}

#[test]
fn reversed_kinematic_joint_reports_load_on_the_authored_child() {
    let rotation = Quat::from_rotation_y(0.5) * Quat::from_rotation_x(-0.4);
    let (mut scene, boundary, dynamic, old) =
        fixture(JointType::Fixed, Vec3::new(32.0, -16.0, 8.0), rotation);
    let old_desc = scene.joint(old).unwrap().clone();
    scene.remove_joint(old).unwrap();
    let joint = scene
        .insert_joint(JointDesc {
            parent: Some(dynamic),
            child: boundary,
            kind: JointType::Fixed,
            axis: Vec3::Z,
            frame_parent: old_desc.frame_child,
            frame_child: old_desc.frame_parent,
        })
        .unwrap();
    assert_ne!(
        scene.model().joint_reversed.host().unwrap()[scene.joint_index(joint).unwrap().index()],
        0
    );
    let mut world = RigidWorld::from_scene(scene).unwrap();
    world.set_reaction_readout(true).unwrap();
    world.step(0.001).unwrap();
    let reading = world.joint_reaction(joint).unwrap();
    near(reading.child.linear, rotation * Vec3::X * 8.0, 1e-9);
    near(reading.parent.linear, rotation * -Vec3::X * 8.0, 1e-9);
    near(reading.child.angular, Vec3::ZERO, 1e-9);
    near(reading.parent.angular, Vec3::ZERO, 1e-9);
}
