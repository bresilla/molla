use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxes, JointAxis, JointMotor, JointSoftness};
use molla_sim::runtime::{BodyDesc, JointDesc, RigidScene};
use molla_sim::{BodyParams, JointTargetMode, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

fn certify(mut make: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let rotation = Quat::from_rotation_z(0.6) * Quat::from_rotation_y(-0.4);
    let authored = Vec3::new(1., 2., -3.).normalize();
    let mut cases = Vec::new();
    for (kind, axis) in [
        (JointType::Revolute, JointAxis::AngX),
        (JointType::Prismatic, JointAxis::LinX),
    ]
    .into_iter()
    .chain([JointAxis::AngX, JointAxis::AngY, JointAxis::AngZ].map(|axis| (JointType::Ball, axis)))
    .chain(JointAxis::ALL.map(|axis| (JointType::D6, axis)))
    {
        for mass in [0.01, 1., 100.] {
            for floating in [false, true] {
                let origin = Vec3::new(3. * cases.len() as f64, 0.5, -0.2);
                let local_direction = if matches!(kind, JointType::Revolute | JointType::Prismatic)
                {
                    authored
                } else {
                    [Vec3::X, Vec3::Y, Vec3::Z][axis as usize % 3]
                };
                let direction = rotation * local_direction;
                let angular = axis as usize >= 3;
                let mut bodies = Vec::new();
                let mut poses = Vec::new();
                let mut velocities = Vec::new();
                let mut masses = Vec::new();
                for i in 0..if floating { 2 } else { 1 } {
                    let parent = floating && i == 0;
                    let body_mass = mass * if parent { 3. } else { 1. };
                    let rate = if parent { -0.03 } else { 0.07 };
                    let pose = Transform::new(origin, rotation);
                    let velocity = if angular {
                        SpatialVector::new(Vec3::ZERO, direction * rate)
                    } else {
                        SpatialVector::new(direction * rate, Vec3::ZERO)
                    };
                    bodies.push(
                        scene
                            .insert_body(BodyDesc {
                                params: BodyParams::new().with_pose(pose),
                                velocity,
                                additional_mass: Some(MassProperties {
                                    mass: body_mass,
                                    local_com: Vec3::ZERO,
                                    inertia: Mat3::IDENTITY * body_mass,
                                }),
                                ..Default::default()
                            })
                            .unwrap(),
                    );
                    masses.push(body_mass);
                    poses.push(pose);
                    velocities.push(rate);
                }
                let parent = floating.then_some(bodies[0]);
                let child = *bodies.last().unwrap();
                let joint = scene
                    .insert_loop_joint(
                        JointDesc {
                            parent,
                            child,
                            kind,
                            axis: authored,
                            frame_parent: if floating {
                                Transform::IDENTITY
                            } else {
                                Transform::new(origin, rotation)
                            },
                            frame_child: Transform::IDENTITY,
                        },
                        JointSoftness {
                            natural_frequency: 0.,
                            damping_ratio: 1.,
                        },
                    )
                    .unwrap();
                if kind == JointType::D6 {
                    scene.set_joint_locked_axes(joint, JointAxes::NONE).unwrap();
                }
                cases.push((
                    joint, kind, axis, mass, direction, bodies, poses, velocities, masses,
                ));
            }
        }
    }
    let mut world = make(scene);
    world.solver.substeps = 1;
    world.set_reaction_readout(true).unwrap();
    #[cfg(feature = "gpu")]
    let mut cached = None;
    let h = 1. / 240.;
    for frame in 0..16 {
        for &(joint, _, axis, mass, _, _, _, _, _) in &cases {
            let stamp = world.scene.model_stamp();
            world
                .scene
                .set_joint_brake(
                    joint,
                    axis,
                    if (4..12).contains(&frame) {
                        0.7 * mass
                    } else {
                        0.
                    },
                )
                .unwrap();
            world
                .scene
                .set_joint_motor(
                    joint,
                    axis,
                    JointMotor {
                        mode: if frame < 8 {
                            JointTargetMode::Effort
                        } else {
                            JointTargetMode::None
                        },
                        target_position: 0.1 * mass,
                        max_force: 0.02 * mass,
                        ..Default::default()
                    },
                )
                .unwrap();
            assert_eq!(world.scene.model_stamp(), stamp);
        }
        world.step(h).unwrap();
        #[cfg(feature = "gpu")]
        if let Some(diagnostics) = world.gpu_diagnostics() {
            let counts = (
                diagnostics.model_compilations,
                diagnostics.current_model_bind_groups_created,
            );
            assert_eq!(*cached.get_or_insert(counts), counts);
        }
        for (joint, kind, axis, mass, direction, bodies, poses, velocities, masses) in &mut cases {
            let child = bodies.len() - 1;
            let parent_rate = if child == 1 { velocities[0] } else { 0. };
            let mobility = masses.iter().map(|mass| 1. / mass).sum::<f64>();
            let brake = if (4..12).contains(&frame) {
                0.7 * *mass
            } else {
                0.
            };
            let motor = if frame < 8 { h * 0.02 * *mass } else { 0. };
            let impulse = (motor - h * brake * (velocities[child] - parent_rate))
                / (1. + h * brake * mobility);
            let label = format!(
                "frame={frame} kind={kind:?} axis={axis:?} mass={mass} bodies={}",
                bodies.len()
            );
            let angular = *axis as usize >= 3;
            for i in 0..bodies.len() {
                velocities[i] += impulse / masses[i] * if i == child { 1. } else { -1. };
                let rate = *direction * velocities[i];
                if angular {
                    poses[i].rotation = (poses[i].rotation
                        + Quat::from_xyzw(rate.x * 0.5, rate.y * 0.5, rate.z * 0.5, 0.)
                            * poses[i].rotation
                            * h)
                        .normalize();
                } else {
                    poses[i].position += rate * h;
                }
                let actual = world.scene.body_velocity(bodies[i]).unwrap();
                let expected = if angular {
                    SpatialVector::new(Vec3::ZERO, rate)
                } else {
                    SpatialVector::new(rate, Vec3::ZERO)
                };
                assert!(
                    (actual.linear - expected.linear).length() < tolerance,
                    "linear {label} body={i}: {actual:?} != {expected:?}"
                );
                assert!(
                    (actual.angular - expected.angular).length() < tolerance,
                    "angular {label} body={i}: {actual:?} != {expected:?}"
                );
                let actual_pose = world.scene.body_pose(bodies[i]).unwrap();
                assert!(
                    (actual_pose.position - poses[i].position).length() < tolerance,
                    "position {label}"
                );
                assert!(
                    (actual_pose.rotation - poses[i].rotation).length() < tolerance,
                    "rotation {label}"
                );
            }
            let load = *direction * (impulse / h);
            let expected = if angular {
                SpatialVector::new(Vec3::ZERO, load)
            } else {
                SpatialVector::new(load, Vec3::ZERO)
            };
            let reading = world.joint_reaction(*joint).unwrap();
            let bound = tolerance * (1. + load.length());
            assert!(
                (reading.child.linear - expected.linear).length() < bound,
                "child force {label}: {:?} != {expected:?}",
                reading.child
            );
            assert!(
                (reading.child.angular - expected.angular).length() < bound,
                "child torque {label}: {:?} != {expected:?}",
                reading.child
            );
            assert!(
                (reading.parent.linear + expected.linear).length() < bound,
                "parent force {label}"
            );
            assert!(
                (reading.parent.angular + expected.angular).length() < bound,
                "parent torque {label}"
            );
        }
    }
    eprintln!(
        "66 rotated compliant hinge/slider/Ball/D6 axis controls: masses .01/1/100, world and floating pairs, capped effort and uncapped physical brakes, 16 live-edit frames, independent rate/pose/both-anchor oracle and retained layout"
    );
}

#[test]
fn cpu_compliant_loop_brakes_cover_all_valid_axes_and_floating_pairs() {
    certify(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn gpu_compliant_loop_brakes_cover_all_valid_axes_and_floating_pairs() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    certify(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-4,
    );
}
