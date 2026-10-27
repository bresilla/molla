use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxes, JointAxis, JointMotor, JointSoftness};
use molla_sim::runtime::{
    BodyDesc, BodyHandle, ColliderDesc, ColliderGeometry, JointDesc, JointHandle, JointSettings,
    RigidScene,
};
use molla_sim::{
    BodyParams, ColliderMass, ContactMaterial, FrictionCoefficients, JointTargetMode, JointType,
    MassProperties,
};
use molla_solvers::rigid_world::RigidWorld;

struct Case {
    body: BodyHandle,
    joint: JointHandle,
    origin: Vec3,
    mass: f64,
    x: f64,
    rate: f64,
    height: f64,
    normal_rate: f64,
    material: ContactMaterial,
    softness: JointSoftness,
    motor: JointMotor,
    brakes: [f64; 2],
    enabled: bool,
}

fn fixture() -> (RigidScene, Vec<Case>) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(-Vec3::Y * 9.81).unwrap();
    let mut cases = Vec::new();
    for mass in [0.01, 1.0, 100.0] {
        for cap in [0.1, f64::INFINITY] {
            let origin = Vec3::new(8.0 * cases.len() as f64, 0.5, 0.0);
            let material = ContactMaterial {
                stiffness: 1.2e4 * mass,
                damping: 10.0 * mass,
                friction: Some(FrictionCoefficients::uniform(0.0)),
                ..Default::default()
            };
            let mut ground = ColliderDesc::new(ColliderGeometry::Cuboid {
                half_extents: Vec3::new(2.0, 0.1, 2.0),
            });
            ground.pose = Transform::from_translation(Vec3::new(origin.x, -0.1, 0.0));
            ground.contact_material = material;
            scene.insert_collider(ground).unwrap();
            let penetration = mass * (9.81 - 0.4_f64.min(cap)) / material.stiffness;
            let body = scene
                .insert_body(BodyDesc {
                    params: BodyParams::new()
                        .with_position(origin + Vec3::X * 0.02 - Vec3::Y * penetration),
                    additional_mass: Some(MassProperties {
                        mass,
                        local_com: Vec3::ZERO,
                        inertia: Mat3::IDENTITY * mass,
                    }),
                    ..Default::default()
                })
                .unwrap();
            let mut shape = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
            shape.parent = Some(body);
            shape.mass = ColliderMass::Mass(0.0);
            shape.contact_material = material;
            scene.insert_collider(shape).unwrap();
            let softness = JointSoftness {
                natural_frequency: 4.0,
                damping_ratio: 0.6,
            };
            let joint = scene
                .insert_loop_joint(
                    JointDesc {
                        parent: None,
                        child: body,
                        kind: JointType::D6,
                        axis: Vec3::X,
                        frame_parent: Transform::from_translation(origin),
                        frame_child: Transform::IDENTITY,
                    },
                    softness,
                )
                .unwrap();
            scene
                .set_joint_locked_axes(joint, JointAxes::NONE.with(JointAxis::LinX))
                .unwrap();
            let motor = JointMotor {
                mode: JointTargetMode::Effort,
                target_position: mass * 0.4,
                max_force: mass * cap,
                ..Default::default()
            };
            scene
                .set_joint_motor(joint, JointAxis::LinY, motor)
                .unwrap();
            scene
                .add_body_wrench(body, SpatialVector::new(Vec3::X * (0.3 * mass), Vec3::ZERO))
                .unwrap();
            cases.push(Case {
                body,
                joint,
                origin,
                mass,
                x: 0.02,
                rate: 0.0,
                height: origin.y - penetration,
                normal_rate: 0.0,
                material,
                softness,
                motor,
                brakes: [0.0; 2],
                enabled: true,
            });
        }
    }
    (scene, cases)
}

fn advance(case: &mut Case, h: f64) -> [f64; 3] {
    let omega = std::f64::consts::TAU * case.softness.natural_frequency;
    let k = case.mass * omega * omega;
    let c = case.mass * 2.0 * case.softness.damping_ratio * omega;
    let brake_factor = 1.0 + h * case.brakes[0] / case.mass;
    let impulse = if case.enabled {
        let spring = (-case.rate / brake_factor - k * case.x / (c + h * k))
            / (1.0 / (case.mass * brake_factor) + 1.0 / (h * (c + h * k)));
        (spring - h * case.brakes[0] * case.rate) / brake_factor
    } else {
        0.0
    };
    case.rate += impulse / case.mass + 0.3 * h;
    case.x += case.rate * h;
    let effort = if case.enabled {
        case.motor
            .target_position
            .clamp(-case.motor.max_force, case.motor.max_force)
    } else {
        0.0
    };
    let normal_impulse = if case.enabled {
        (h * effort - h * case.brakes[1] * case.normal_rate)
            / (1.0 + h * case.brakes[1] / case.mass)
    } else {
        0.0
    };
    let depth = 0.5 - case.height;
    assert!(
        depth > 0.0 && depth < 0.002,
        "fixture contact depth={depth}"
    );
    let pressure =
        case.material.stiffness * depth + case.material.damping * (-case.normal_rate).max(0.0);
    case.normal_rate += normal_impulse / case.mass + h * (pressure / case.mass - 9.81);
    case.height += case.normal_rate * h;
    [impulse / h, normal_impulse / h, pressure]
}

fn certify(mut make_world: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64, brakes: bool) {
    for hz in [60, 240] {
        for substeps in [1, 4] {
            let (scene, mut cases) = fixture();
            let mut world = make_world(scene);
            world.solver.substeps = substeps;
            world.set_reaction_readout(true).unwrap();
            for frame in 0..20 {
                for case in &mut cases {
                    if brakes {
                        case.brakes = if (4..19).contains(&frame) {
                            [12.0 * case.mass, 7.0 * case.mass]
                        } else {
                            [0.0; 2]
                        };
                        world
                            .scene
                            .set_joint_brake(case.joint, JointAxis::LinX, case.brakes[0])
                            .unwrap();
                        world
                            .scene
                            .set_joint_brake(case.joint, JointAxis::LinY, case.brakes[1])
                            .unwrap();
                    }
                    if frame == 4 {
                        case.softness.natural_frequency = 7.0;
                        world
                            .scene
                            .set_joint_softness(case.joint, Some(case.softness))
                            .unwrap();
                    }
                    if frame == 8 {
                        case.motor.max_force = 0.0;
                    }
                    if frame == 12 {
                        case.motor.target_position = case.mass * 0.2;
                        case.motor.max_force = 0.3 * case.mass;
                    }
                    if frame == 8 || frame == 12 {
                        world
                            .scene
                            .set_joint_motor(case.joint, JointAxis::LinY, case.motor)
                            .unwrap();
                    }
                    if frame == 16 || frame == 18 {
                        case.enabled = frame == 18;
                        world
                            .scene
                            .set_joint_settings(
                                case.joint,
                                JointSettings {
                                    enabled: case.enabled,
                                    ..Default::default()
                                },
                            )
                            .unwrap();
                    }
                }
                let mut forces = vec![[0.0; 3]; cases.len()];
                for _ in 0..substeps {
                    for (case, force) in cases.iter_mut().zip(&mut forces) {
                        let sample = advance(case, 1.0 / (hz as f64 * substeps as f64));
                        for axis in 0..3 {
                            force[axis] += sample[axis] / substeps as f64;
                        }
                    }
                }
                world.step(1.0 / hz as f64).unwrap_or_else(|error| {
                    panic!("contact loop hz={hz} substeps={substeps} frame={frame}: {error}")
                });
                let contacts = world.contact_pairs();
                assert_eq!(contacts.len(), cases.len());
                for (case, force) in cases.iter().zip(forces) {
                    let pose = world.scene.body_pose(case.body).unwrap();
                    let velocity = world.scene.body_velocity(case.body).unwrap();
                    assert!(
                        (pose.position.x - case.origin.x - case.x).abs() < tolerance,
                        "contact loop x hz={hz} substeps={substeps} frame={frame} mass={} expected={} actual={pose:?}",
                        case.mass,
                        case.x
                    );
                    assert!((velocity.linear.x - case.rate).abs() < tolerance);
                    assert!(
                        (pose.position.y - case.height).abs() < tolerance,
                        "contact loop height hz={hz} substeps={substeps} frame={frame} mass={} expected={} pose={pose:?}",
                        case.mass,
                        case.height
                    );
                    assert!(
                        (velocity.linear.y - case.normal_rate).abs() < tolerance,
                        "contact loop normal rate hz={hz} substeps={substeps} frame={frame} mass={} expected={} actual={velocity:?}",
                        case.mass,
                        case.normal_rate
                    );
                    assert!(velocity.angular.length() < tolerance);
                    let pair = contacts
                        .iter()
                        .find(|pair| pair.bodies.contains(&Some(case.body)))
                        .unwrap();
                    let normal_impulse: f64 =
                        pair.points.iter().map(|point| point.normal_impulse).sum();
                    let expected_impulse = force[2] / hz as f64;
                    assert!(
                        (normal_impulse - expected_impulse).abs()
                            < tolerance * (1.0 + expected_impulse),
                        "contact loop pressure hz={hz} substeps={substeps} frame={frame} mass={} expected={expected_impulse} actual={normal_impulse}",
                        case.mass
                    );
                    if case.enabled {
                        let expected = Vec3::new(force[0], force[1], 0.0);
                        let reaction = world.joint_reaction(case.joint).unwrap();
                        assert!(
                            (reaction.child.linear - expected).length()
                                < tolerance * (1.0 + expected.length()),
                            "contact loop reaction hz={hz} substeps={substeps} frame={frame} expected={expected:?} actual={reaction:?}"
                        );
                        assert!(reaction.child.angular.length() < tolerance);
                        assert!(
                            (reaction.parent.linear + reaction.child.linear).length() < tolerance
                        );
                    } else {
                        assert!(world.joint_reaction(case.joint).is_none());
                    }
                }
                assert!(world.quarantined_bodies().is_empty());
            }
        }
    }
}

#[test]
fn cpu_compliant_loops_and_capped_effort_preserve_contact_and_horizontal_recurrence() {
    certify(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8, false);
}

#[test]
fn cpu_compliant_brakes_share_live_contacts_and_retire_with_disabled_loops() {
    certify(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8, true);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn gpu_compliant_brakes_share_live_contacts_and_retire_with_disabled_loops() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    certify(
        |scene| {
            RigidWorld::from_scene_gpu_with_contact_capacity(
                scene,
                engine.clone(),
                molla_solvers::rigid_world::GpuContactCapacity {
                    pairs: 32,
                    contacts_per_island: 8,
                },
            )
            .unwrap()
        },
        2e-4,
        true,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn gpu_compliant_loops_and_capped_effort_preserve_contact_and_horizontal_recurrence() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    certify(
        |scene| {
            RigidWorld::from_scene_gpu_with_contact_capacity(
                scene,
                engine.clone(),
                molla_solvers::rigid_world::GpuContactCapacity {
                    pairs: 32,
                    contacts_per_island: 8,
                },
            )
            .unwrap()
        },
        2e-4,
        false,
    );
}
