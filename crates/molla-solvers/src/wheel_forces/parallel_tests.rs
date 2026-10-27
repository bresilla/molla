use super::{SERIAL_TYRES, WheelDesc};
use crate::pressure_tire::PressureTireConfig;
use crate::rigid_world::RigidWorld;
use molla_math::{Mat3, Transform, Vec3};
use molla_sim::joint_control::{JointAxis, JointMotor};
use molla_sim::runtime::{BodyDesc, BodyHandle, ColliderDesc, ColliderGeometry, JointDesc};
use molla_sim::{BodyParams, JointTargetMode, JointType, MassProperties};
use molla_vehicle::RigidTire;
use molla_vehicle::real_tire::RealTireParams;
use std::sync::Arc;

fn body(world: &mut RigidWorld, point: Vec3, mass: f64, inertia: Vec3) -> BodyHandle {
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

/// Three four-wheeled carts on brush tyres, driven at different speeds.
fn carts() -> RigidWorld {
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
    for cart in 0..3 {
        let origin = Vec3::new(0.0, 1.34, cart as f64 * 6.0);
        let chassis = body(&mut world, origin, 3774.0, Vec3::new(3606.0, 453.0, 3703.0));
        for x in [-1.4, 1.4] {
            for z in [-1.0, 1.0] {
                let offset = Vec3::new(x, -0.7, z);
                let wheel = body(&mut world, origin + offset, 200.0, Vec3::splat(40.0));
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
                let motor = JointMotor {
                    mode: JointTargetMode::Velocity,
                    target_velocity: (1.0 + cart as f64 + x * 0.1) / 0.69,
                    damping: 2000.0,
                    max_force: 5000.0,
                    ..Default::default()
                };
                world
                    .scene
                    .set_joint_motor(spin, JointAxis::AngX, motor)
                    .unwrap();
                let mut collider = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.69 });
                collider.parent = Some(wheel);
                world.scene.insert_collider(collider).unwrap();
                let desc = WheelDesc {
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
                };
                world.wheels.insert(&world.scene, desc).unwrap();
                let config = PressureTireConfig {
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
                };
                world.wheels.configure_pressure_tire(wheel, config).unwrap();
            }
        }
    }
    world
}

/// The bits of every pose, velocity and tyre force after each step.
fn trace(serial: bool) -> Vec<u64> {
    SERIAL_TYRES.set(serial);
    let mut world = carts();
    let mut bits = Vec::new();
    for _ in 0..120 {
        world.step(1.0 / 120.0).unwrap();
        let samples = world.wheels.samples(&world.scene);
        assert!(samples.len() >= 12);
        assert!(
            samples
                .iter()
                .all(|s| s.pressure.is_some() && s.output.in_contact)
        );
        let state = world.scene.state();
        for pose in state.body_q.host().unwrap() {
            bits.extend(pose.position.to_array().map(f64::to_bits));
            bits.extend(pose.rotation.to_array().map(f64::to_bits));
        }
        for velocity in state.body_qd.host().unwrap() {
            bits.extend(velocity.linear.to_array().map(f64::to_bits));
            bits.extend(velocity.angular.to_array().map(f64::to_bits));
        }
        for sample in world.wheels.samples(&world.scene).iter() {
            bits.extend(sample.output.force.map(f64::to_bits));
        }
    }
    SERIAL_TYRES.set(false);
    bits
}

#[test]
fn brush_tyres_step_the_same_bits_on_the_pool_as_one_by_one() {
    assert!(trace(true) == trace(false));
}
