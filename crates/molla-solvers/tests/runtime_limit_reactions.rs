use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::JointAxis;
use molla_sim::runtime::{BodyDesc, JointDesc};
use molla_sim::{JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

#[test]
fn scalar_joint_stops_transfer_load_without_external_momentum() {
    for kind in [JointType::Prismatic, JointType::Revolute] {
        for sign in [-1.0, 1.0] {
            let mut world = RigidWorld::new().unwrap();
            world.scene.set_gravity(Vec3::ZERO).unwrap();
            let desc = BodyDesc {
                additional_mass: Some(MassProperties {
                    mass: 2.0,
                    local_com: Vec3::ZERO,
                    inertia: Mat3::from_diagonal(Vec3::splat(4.0)),
                }),
                ..Default::default()
            };
            let a = world.scene.insert_body(desc.clone()).unwrap();
            let b = world.scene.insert_body(desc).unwrap();
            let joint = world
                .scene
                .insert_joint(JointDesc {
                    parent: Some(a),
                    child: b,
                    kind,
                    axis: Vec3::X,
                    frame_parent: Transform::IDENTITY,
                    frame_child: Transform::IDENTITY,
                })
                .unwrap();
            let axis = if kind == JointType::Prismatic {
                JointAxis::LinX
            } else {
                JointAxis::AngX
            };
            world
                .scene
                .set_joint_limits(
                    joint,
                    axis,
                    Some(if sign < 0.0 { [0.0, 1.0] } else { [-1.0, 0.0] }),
                )
                .unwrap();
            let force = Vec3::X * (8.0 * sign);
            let wrench = if kind == JointType::Prismatic {
                SpatialVector::new(force, Vec3::ZERO)
            } else {
                SpatialVector::new(Vec3::ZERO, force)
            };
            world.scene.add_body_wrench(b, wrench).unwrap();
            world.step(0.1).unwrap();
            let va = world.scene.body_velocity(a).unwrap();
            let vb = world.scene.body_velocity(b).unwrap();
            let (actual, expected) = if kind == JointType::Prismatic {
                ((va.linear + vb.linear) * 2.0, force * 0.1)
            } else {
                ((va.angular + vb.angular) * 4.0, force * 0.1)
            };
            assert!(
                (actual - expected).length() < 1e-8,
                "{kind:?} sign={sign}: momentum {actual:?} != {expected:?}"
            );
            let index = world.scene.joint_index(joint).unwrap().index();
            let offset = world.scene.model().joint_coord_offset.host().unwrap()[index] as usize;
            assert!(world.scene.state().joint_q.host().unwrap()[offset].abs() < 1e-10);
            let relative = if kind == JointType::Prismatic {
                vb.linear - va.linear
            } else {
                vb.angular - va.angular
            };
            assert!(relative.length() < 1e-10, "stop did not hold");
            world.scene.reset_body_forces(b).unwrap();
            world
                .scene
                .add_body_wrench(b, SpatialVector::new(-wrench.linear, -wrench.angular))
                .unwrap();
            world.step(0.1).unwrap();
            let q = world.scene.state().joint_q.host().unwrap()[offset];
            assert!(q * sign < -1e-4, "unilateral stop did not release: {q}");
        }
    }
}

#[test]
fn coupled_prismatic_stops_preserve_momentum_and_dissipate_impact() {
    for masses in [[2.0, 3.0, 5.0], [1000.0, 1.0, 1000.0]] {
        for shift in [Vec3::ZERO, Vec3::new(30.0, -10.0, 5.0)] {
            let mut world = RigidWorld::new().unwrap();
            world.scene.set_gravity(Vec3::ZERO).unwrap();
            let mut bodies = Vec::new();
            for (mass, vx) in masses.into_iter().zip([1.0, -2.0, -4.0]) {
                bodies.push(
                    world
                        .scene
                        .insert_body(BodyDesc {
                            params: molla_sim::BodyParams {
                                pose: Transform::from_translation(shift),
                                ..Default::default()
                            },
                            velocity: SpatialVector::new(Vec3::X * vx, Vec3::ZERO),
                            additional_mass: Some(MassProperties {
                                mass,
                                local_com: Vec3::ZERO,
                                inertia: Mat3::IDENTITY,
                            }),
                            ..Default::default()
                        })
                        .unwrap(),
                );
            }
            let mut joints = Vec::new();
            for pair in bodies.windows(2) {
                let j = world
                    .scene
                    .insert_joint(JointDesc {
                        parent: Some(pair[0]),
                        child: pair[1],
                        kind: JointType::Prismatic,
                        axis: Vec3::X,
                        frame_parent: Transform::IDENTITY,
                        frame_child: Transform::IDENTITY,
                    })
                    .unwrap();
                world
                    .scene
                    .set_joint_limits(j, JointAxis::LinX, Some([0.0, 1.0]))
                    .unwrap();
                joints.push(j);
            }
            let momentum = |w: &RigidWorld| {
                bodies
                    .iter()
                    .map(|b| {
                        w.scene.body_velocity(*b).unwrap().linear
                            * w.scene.body_mass_properties(*b).unwrap().mass
                    })
                    .sum::<Vec3>()
            };
            let energy = |w: &RigidWorld| {
                bodies
                    .iter()
                    .map(|b| {
                        0.5 * w.scene.body_mass_properties(*b).unwrap().mass
                            * w.scene.body_velocity(*b).unwrap().linear.length_squared()
                    })
                    .sum::<f64>()
            };
            let before = momentum(&world);
            let initial_energy = energy(&world);
            world.step(1.0 / 120.0).unwrap();
            assert!((momentum(&world) - before).length() < 1e-8);
            assert!(energy(&world) < initial_energy - 1.0);
            for b in &bodies {
                assert!(
                    (world.scene.body_velocity(*b).unwrap().linear.x
                        - before.x / masses.iter().sum::<f64>())
                    .abs()
                        < 1e-8
                );
            }
            for j in joints {
                let index = world.scene.joint_index(j).unwrap().index();
                let offset = world.scene.model().joint_coord_offset.host().unwrap()[index] as usize;
                assert!(world.scene.state().joint_q.host().unwrap()[offset].abs() < 1e-9);
            }
        }
    }
}
