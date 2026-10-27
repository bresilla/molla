use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxes, JointAxis, JointMotor, MotorModel};
use molla_sim::runtime::{BodyDesc, JointDesc, RigidScene};
use molla_sim::{BodyParams, JointTargetMode, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

fn certify(mut make: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let mut cases = Vec::new();
    let rotation = Quat::from_rotation_z(0.6) * Quat::from_rotation_x(-0.4);
    let authored = Vec3::new(1., 2., -3.).normalize();
    let alignment = Quat::from_rotation_arc(Vec3::X, authored);
    let frame = (rotation * alignment).normalize();
    let com = Vec3::new(0.12, 0.07, -0.05);
    let anchor = Vec3::new(0.3, -0.2, 0.1);
    let arm = rotation * (com - anchor);
    let local_inertia = Mat3::from_cols(
        Vec3::new(1.9, 0.23, -0.17),
        Vec3::new(0.23, 1.6, 0.11),
        Vec3::new(-0.17, 0.11, 1.4),
    );
    let h = 1. / 240.;
    for (kind, axis) in [
        (JointType::Revolute, JointAxis::AngX),
        (JointType::Prismatic, JointAxis::LinX),
    ]
    .into_iter()
    .chain(JointAxis::ALL.map(|axis| (JointType::D6, axis)))
    {
        for mass in [0.01, 1., 100.] {
            for model in [MotorModel::Force, MotorModel::Acceleration] {
                for cap in [f64::INFINITY, 0.2 * mass, 0.] {
                    let position = Vec3::new(3. * cases.len() as f64, 0.5, -0.2);
                    let pose = Transform::new(position, rotation);
                    let body = scene
                        .insert_body(BodyDesc {
                            params: BodyParams::new().with_pose(pose),
                            additional_mass: Some(MassProperties {
                                mass,
                                local_com: com,
                                inertia: local_inertia * mass,
                            }),
                            ..Default::default()
                        })
                        .unwrap();
                    let child_frame = Transform::new(
                        anchor,
                        if kind == JointType::D6 {
                            alignment
                        } else {
                            Quat::IDENTITY
                        },
                    );
                    let parent_frame = pose.compose(&child_frame);
                    let joint = scene
                        .insert_rigid_loop_joint(JointDesc {
                            parent: None,
                            child: body,
                            kind,
                            axis: authored,
                            frame_parent: parent_frame,
                            frame_child: child_frame,
                        })
                        .unwrap();
                    if kind == JointType::D6 {
                        scene
                            .set_joint_locked_axes(joint, JointAxes(63 & !(1 << axis as u8)))
                            .unwrap();
                    }
                    let direction = frame * [Vec3::X, Vec3::Y, Vec3::Z][axis as usize % 3];
                    let angular = axis as usize >= 3;
                    let inertia = Mat3::from_quat(rotation)
                        * local_inertia
                        * mass
                        * Mat3::from_quat(rotation).transpose();
                    let effective = if angular {
                        direction.dot(inertia * direction)
                            + mass * direction.cross(arm).length_squared()
                    } else {
                        mass
                    };
                    let motor = JointMotor {
                        mode: JointTargetMode::PositionVelocity,
                        target_position: 0.003,
                        target_velocity: 0.1,
                        stiffness: 30.,
                        damping: 3.,
                        max_force: cap,
                        model,
                    };
                    scene.set_joint_motor(joint, axis, motor).unwrap();
                    let scale = if model == MotorModel::Acceleration {
                        effective
                    } else {
                        1.
                    };
                    let kp = motor.stiffness * scale;
                    let kd = motor.damping * scale;
                    let impulse = (h * (kp * motor.target_position + kd * motor.target_velocity)
                        / (1. + h * (kd + h * kp) / effective))
                        .clamp(-h * cap, h * cap);
                    let rate = impulse / effective;
                    let spin = if angular {
                        direction * rate
                    } else {
                        Vec3::ZERO
                    };
                    let linear = if angular {
                        direction.cross(arm) * rate
                    } else {
                        direction * rate
                    };
                    let next_rotation = (rotation
                        + Quat::from_xyzw(spin.x * 0.5, spin.y * 0.5, spin.z * 0.5, 0.)
                            * rotation
                            * h)
                        .normalize();
                    let next_inertia = Mat3::from_quat(next_rotation)
                        * local_inertia
                        * mass
                        * Mat3::from_quat(next_rotation).transpose();
                    let expected_rate =
                        SpatialVector::new(linear, next_inertia.inverse() * (inertia * spin));
                    let expected_pose = Transform::new(
                        position + rotation * com + linear * h - next_rotation * com,
                        next_rotation,
                    );
                    let force = linear * (mass / h);
                    let child = SpatialVector::new(force, inertia * spin / h + arm.cross(force));
                    let parent = child.scale(-1.);
                    cases.push((
                        body,
                        joint,
                        mass,
                        model,
                        axis,
                        cap,
                        effective,
                        expected_rate,
                        expected_pose,
                        child,
                        parent,
                    ));
                }
            }
        }
    }
    let mut world = make(scene);
    world.solver.substeps = 1;
    world.set_reaction_readout(true).unwrap();
    world.step(h).unwrap();
    for (body, joint, mass, model, axis, cap, effective, rate, pose, child, parent) in cases {
        let label =
            format!("axis={axis:?} model={model:?} mass={mass} cap={cap} effective={effective}");
        let actual = world.scene.body_velocity(body).unwrap();
        assert!(
            (actual.linear - rate.linear).length() < tolerance,
            "linear {label}: {actual:?} != {rate:?}"
        );
        assert!(
            (actual.angular - rate.angular).length() < tolerance,
            "angular {label}: {actual:?} != {rate:?}"
        );
        let actual = world.scene.body_pose(body).unwrap();
        assert!(
            (actual.position - pose.position).length() < tolerance,
            "position {label}"
        );
        assert!(
            (actual.rotation - pose.rotation).length() < tolerance,
            "rotation {label}"
        );
        let reading = world.joint_reaction(joint).unwrap();
        let bound = tolerance * (1. + mass + child.linear.length() + child.angular.length());
        assert!(
            (reading.child.linear - child.linear).length() < bound,
            "child force {label}: {:?} != {child:?}",
            reading.child
        );
        assert!(
            (reading.child.angular - child.angular).length() < bound,
            "child torque {label}: {:?} != {child:?}",
            reading.child
        );
        assert!(
            (reading.parent.linear - parent.linear).length() < bound,
            "parent force {label}"
        );
        assert!(
            (reading.parent.angular - parent.angular).length() < bound,
            "parent torque {label}: {:?} != {parent:?}",
            reading.parent
        );
    }
    eprintln!(
        "144 rotated off-centre/full-inertia hinge/slider/D6 closures: Force/Acceleration, masses .01/1/100, uncapped/capped/zero, independent pivot inertia and both-anchor reactions"
    );
}

#[test]
fn cpu_acceleration_drives_use_locked_pivot_inertia_and_physical_force_caps() {
    certify(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; use make test-gpu"]
fn gpu_acceleration_drives_use_locked_pivot_inertia_and_physical_force_caps() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    certify(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-3,
    );
}

fn nulls(mut make: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let rotation = Quat::from_rotation_z(0.6) * Quat::from_rotation_x(-0.4);
    let direction = rotation * Vec3::X;
    let mut cases = Vec::new();
    for duplicate in [false, true] {
        for mass in [0.01, 1., 100.] {
            let pose = Transform::new(Vec3::X * (cases.len() as f64 * 3.), rotation);
            let body = scene
                .insert_body(BodyDesc {
                    params: BodyParams::new().with_pose(pose),
                    additional_mass: Some(MassProperties {
                        mass,
                        local_com: Vec3::ZERO,
                        inertia: Mat3::IDENTITY * mass,
                    }),
                    ..Default::default()
                })
                .unwrap();
            let desc = JointDesc {
                parent: None,
                child: body,
                kind: JointType::D6,
                axis: Vec3::X,
                frame_parent: pose,
                frame_child: Transform::IDENTITY,
            };
            let joint = scene.insert_rigid_loop_joint(desc.clone()).unwrap();
            scene
                .set_joint_locked_axes(
                    joint,
                    if duplicate {
                        JointAxes(62)
                    } else {
                        JointAxes::ALL
                    },
                )
                .unwrap();
            scene
                .set_joint_motor(
                    joint,
                    JointAxis::LinX,
                    JointMotor {
                        mode: JointTargetMode::PositionVelocity,
                        target_position: 0.003,
                        target_velocity: 0.1,
                        stiffness: 30.,
                        damping: 3.,
                        model: MotorModel::Acceleration,
                        ..Default::default()
                    },
                )
                .unwrap();
            let fixed = if duplicate {
                Some(
                    scene
                        .insert_rigid_loop_joint(JointDesc {
                            kind: JointType::Fixed,
                            ..desc
                        })
                        .unwrap(),
                )
            } else {
                None
            };
            let force = direction * (0.1 * mass);
            scene
                .add_body_wrench(body, SpatialVector::new(force, Vec3::ZERO))
                .unwrap();
            cases.push((body, joint, fixed, pose, mass, force));
        }
    }
    let mut world = make(scene);
    world.solver.substeps = 1;
    world.set_reaction_readout(true).unwrap();
    world.step(1. / 240.).unwrap();
    for (body, joint, fixed, pose, mass, force) in cases {
        let actual = world.scene.body_pose(body).unwrap();
        assert!((actual.position - pose.position).length() < tolerance);
        assert!((actual.rotation - pose.rotation).length() < tolerance);
        let rate = world.scene.body_velocity(body).unwrap();
        assert!(rate.linear.length() < tolerance && rate.angular.length() < tolerance);
        let reaction = world.joint_reaction(joint).unwrap();
        let bound = tolerance * (1. + mass);
        assert!(
            (reaction.child.linear + if fixed.is_some() { Vec3::ZERO } else { force }).length()
                < bound,
            "null drive force mass={mass} duplicate={fixed:?}: {:?}",
            reaction.child
        );
        assert!(reaction.child.angular.length() < bound);
        if let Some(fixed) = fixed {
            let support = world.joint_reaction(fixed).unwrap();
            assert!((support.child.linear + force).length() < bound);
            assert!(support.child.angular.length() < bound);
        }
    }
    eprintln!(
        "zero conditional mobility: locked D6 and redundant Fixed/D6 rows, masses .01/1/100, controls retire while hard support remains physical"
    );
}

#[test]
fn cpu_zero_acceleration_mobility_retires_drives_and_preserves_redundant_hard_support() {
    nulls(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; use make test-gpu"]
fn gpu_zero_acceleration_mobility_retires_drives_and_preserves_redundant_hard_support() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    nulls(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-3,
    );
}
