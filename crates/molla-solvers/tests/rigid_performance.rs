//! Synthetic pressure-tyre CPU workload, separate from imported-machine acceptance.

use std::sync::Arc;
use std::time::Instant;

use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxes, JointAxis, JointMotor, JointSoftness};
use molla_sim::runtime::{BodyDesc, BodyHandle, ColliderDesc, ColliderGeometry, JointDesc};
use molla_sim::{BodyParams, JointTargetMode, JointType, MassProperties};
use molla_solvers::pressure_tire::PressureTireConfig;
use molla_solvers::rigid_world::RigidWorld;
use molla_solvers::wheel_forces::WheelDesc;
use molla_vehicle::RigidTire;
use molla_vehicle::real_tire::RealTireParams;

const DT: f64 = 1.0 / 120.0;

fn add_body(world: &mut RigidWorld, point: Vec3, mass: f64, inertia: Vec3) -> BodyHandle {
    world
        .scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_pose(Transform::from_translation(point)),
            additional_mass: Some(MassProperties {
                mass,
                local_com: Vec3::ZERO,
                inertia: Mat3::from_diagonal(inertia),
            }),
            ..Default::default()
        })
        .unwrap()
}

fn fixture(driving: bool) -> RigidWorld {
    let mut world = RigidWorld::new().unwrap();
    let mut floor = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(100.0, 0.1, 100.0),
    });
    floor.pose = Transform::from_translation(-Vec3::Y * 0.1);
    let ground = world.scene.insert_collider(floor).unwrap();
    world
        .wheels
        .register_ground(&world.scene, ground, None)
        .unwrap();
    let origin = Vec3::Y * 1.34;
    let chassis = add_body(&mut world, origin, 3774.0, Vec3::new(3606.0, 453.0, 3703.0));
    for x in [-1.4, 1.4] {
        for z in [-1.0, 1.0] {
            let offset = Vec3::new(x, -0.7, z);
            let wheel = add_body(&mut world, origin + offset, 200.0, Vec3::splat(40.0));
            let spin = world
                .scene
                .insert_joint(JointDesc {
                    parent: Some(chassis),
                    child: wheel,
                    kind: JointType::Revolute,
                    frame_parent: Transform::from_translation(offset),
                    frame_child: Transform::IDENTITY,
                    axis: -Vec3::Z,
                })
                .unwrap();
            if driving {
                world
                    .scene
                    .set_joint_motor(
                        spin,
                        JointAxis::AngX,
                        JointMotor {
                            mode: JointTargetMode::Velocity,
                            target_velocity: 2.0 / 0.69,
                            damping: 2000.0,
                            max_force: 5000.0,
                            ..Default::default()
                        },
                    )
                    .unwrap();
            }
            let mut collider = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.69 });
            collider.parent = Some(wheel);
            world.scene.insert_collider(collider).unwrap();
            world
                .wheels
                .insert(
                    &world.scene,
                    WheelDesc {
                        body: wheel,
                        spin_joint: spin,
                        heading_body: chassis,
                        local_heading: Vec3::X,
                        local_hub: Vec3::ZERO,
                        spin_sign: 1.0,
                        radius: 0.69,
                        normal_stiffness: 100_000.0,
                        normal_damping: 1000.0,
                        tire: Arc::new(RigidTire::default()),
                    },
                )
                .unwrap();
            world
                .wheels
                .configure_pressure_tire(
                    wheel,
                    PressureTireConfig {
                        tire: RealTireParams {
                            radius: 0.69,
                            width: 0.42,
                            pressure_pa: 180_000.0,
                            ..Default::default()
                        },
                        min_pressure_pa: 50_000.0,
                        max_pressure_pa: 400_000.0,
                        pressure_rate_pa_s: 100_000.0,
                        supported_mass: 4910.0 / 4.0,
                        damping_ratio: 0.7,
                        hysteresis_fraction: 0.1,
                    },
                )
                .unwrap();
        }
    }
    for i in 0..21 {
        let offset = Vec3::new((i % 7) as f64 * 0.1 - 0.3, 0.2, 0.0);
        let body = add_body(&mut world, origin + offset, 16.0, Vec3::splat(2.0));
        let joint = world
            .scene
            .insert_joint(JointDesc {
                parent: Some(chassis),
                child: body,
                kind: if i == 0 {
                    JointType::D6
                } else {
                    JointType::Fixed
                },
                frame_parent: Transform::from_translation(offset),
                frame_child: Transform::IDENTITY,
                axis: Vec3::Y,
            })
            .unwrap();
        if i == 0 {
            world
                .scene
                .set_joint_locked_axes(
                    joint,
                    JointAxes::LIN.with(JointAxis::AngX).with(JointAxis::AngZ),
                )
                .unwrap();
        }
    }
    world
}

fn assert_supported(world: &RigidWorld) {
    assert_eq!(world.scene.model().body_count, 26);
    assert!(world.quarantined_bodies().is_empty());
    let samples = world.wheels.samples(&world.scene);
    assert_eq!(samples.len(), 32);
    assert!(
        samples
            .iter()
            .all(|s| s.output.in_contact && s.output.fz.is_finite() && s.pressure.is_some())
    );
    let force = samples.iter().map(|s| s.output.fz * s.dt / DT).sum::<f64>();
    assert!((force - 4910.0 * 9.81).abs() < 491.0, "support={force}");
}

fn loop_fixture() -> RigidWorld {
    let mut world = fixture(true);
    let (joint, desc) = world
        .scene
        .joints()
        .find(|(_, desc)| desc.kind == JointType::D6)
        .map(|(id, desc)| (id, desc.clone()))
        .unwrap();
    world
        .scene
        .set_joint_motor(
            joint,
            JointAxis::AngY,
            JointMotor {
                mode: JointTargetMode::Position,
                target_position: 0.1,
                stiffness: 1000.0,
                damping: 100.0,
                max_force: 500.0,
                ..Default::default()
            },
        )
        .unwrap();
    let mut closure = desc;
    closure.kind = JointType::Fixed;
    closure.frame_parent.rotation = Quat::from_rotation_y(0.1);
    world
        .scene
        .insert_loop_joint(
            closure,
            JointSoftness {
                natural_frequency: 20.0,
                damping_ratio: 1.0,
            },
        )
        .unwrap();
    world
}

#[test]
fn pressure_loop_fixture_replays_and_converges() {
    let mut a = loop_fixture();
    let mut b = loop_fixture();
    for _ in 0..240 {
        for world in [&mut a, &mut b] {
            world
                .step_filtered(DT, |a, b| a.is_some() && b.is_some())
                .unwrap();
        }
    }
    assert_supported(&a);
    assert_supported(&b);
    assert_eq!(
        a.scene.state().body_q.host().unwrap(),
        b.scene.state().body_q.host().unwrap()
    );
    assert_eq!(
        a.scene.state().body_qd.host().unwrap(),
        b.scene.state().body_qd.host().unwrap()
    );
    let (_, joint) = a.scene.loop_joints().next().unwrap();
    let parent = a
        .scene
        .body_pose(joint.parent.unwrap())
        .unwrap()
        .compose(&joint.frame_parent);
    let child = a
        .scene
        .body_pose(joint.child)
        .unwrap()
        .compose(&joint.frame_child);
    assert!(parent.rotation.angle_between(child.rotation) < 1e-5);
    assert!(a.scene.state().body_qd.host().unwrap()[0].linear.x > 1.0);
}

#[test]
#[ignore = "wall-clock CPU benchmark; run in release without concurrent builds"]
fn pressure_loop_motor_step_timings() {
    let mut world = loop_fixture();
    for _ in 0..240 {
        world
            .step_filtered(DT, |a, b| a.is_some() && b.is_some())
            .unwrap();
    }
    for batch in 0..5 {
        let start = Instant::now();
        for _ in 0..240 {
            world
                .step_filtered(DT, |a, b| a.is_some() && b.is_some())
                .unwrap();
        }
        eprintln!(
            "pressure loop/motor fixture batch {batch}: {:.6} ms/step",
            start.elapsed().as_secs_f64() * 1000.0 / 240.0
        );
        assert_supported(&world);
    }
}

#[test]
fn pressure_fixture_stays_supported_and_replays_exactly() {
    let mut a = fixture(true);
    let mut b = fixture(true);
    for _ in 0..240 {
        a.step_filtered(DT, |a, b| a.is_some() && b.is_some())
            .unwrap();
        b.step_filtered(DT, |a, b| a.is_some() && b.is_some())
            .unwrap();
    }
    assert_supported(&a);
    assert_supported(&b);
    assert!(a.scene.state().body_qd.host().unwrap()[0].linear.x > 1.0);
    assert_eq!(
        a.scene.state().body_q.host().unwrap(),
        b.scene.state().body_q.host().unwrap()
    );
    assert_eq!(
        a.scene.state().body_qd.host().unwrap(),
        b.scene.state().body_qd.host().unwrap()
    );
    assert!(
        a.scene
            .state()
            .body_f
            .host()
            .unwrap()
            .iter()
            .all(|f| *f == SpatialVector::ZERO)
    );
}

#[test]
#[ignore = "wall-clock CPU benchmark; run in release without concurrent builds"]
fn pressure_enabled_26_body_step_timings() {
    for driving in [false, true] {
        let mut world = fixture(driving);
        for _ in 0..240 {
            world
                .step_filtered(DT, |a, b| a.is_some() && b.is_some())
                .unwrap();
        }
        assert_supported(&world);
        if driving {
            assert!(world.scene.state().body_qd.host().unwrap()[0].linear.x > 1.0);
        }
        for batch in 0..5 {
            let start = Instant::now();
            for _ in 0..240 {
                world
                    .step_filtered(DT, |a, b| a.is_some() && b.is_some())
                    .unwrap();
            }
            eprintln!(
                "pressure fixture driving={driving} batch {batch}: {:.6} ms/step, 26 bodies, 4 tyres, 8 substeps",
                start.elapsed().as_secs_f64() * 1000.0 / 240.0
            );
            assert_supported(&world);
        }
    }
}
