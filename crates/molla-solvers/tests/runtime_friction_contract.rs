use molla_math::{Mat3, Quat, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, BodyHandle, ColliderDesc, ColliderGeometry, RigidScene};
use molla_sim::{BodyParams, ColliderMass, ContactMaterial, FrictionCoefficients, MassProperties};
use molla_solvers::{penalty_contact::FrictionRegime, rigid_world::RigidWorld};

struct Case {
    body: BodyHandle,
    normal: Vec3,
    start: Vec3,
    settled: Vec3,
    peak: f64,
    creep: f64,
}

fn fixture() -> (RigidScene, Vec<Case>) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(-Vec3::Y * 10.0).unwrap();
    let mut cases = Vec::new();
    let slope: f64 = 0.3;
    for mass in [1.0, 100.0] {
        for yaw in [0.0, 0.7] {
            let rotation = Quat::from_rotation_y(yaw) * Quat::from_rotation_z(slope);
            let normal = rotation * Vec3::Y;
            let offset = Vec3::new(100.0 + 16.0 * cases.len() as f64, -30.0, 20.0);
            let stiffness = 5e4 * mass;
            let penetration = mass * 10.0 * slope.cos() / (4.0 * stiffness);
            let start = offset + normal * (0.5 - penetration);
            let body = scene
                .insert_body(BodyDesc {
                    params: BodyParams::new().with_pose(Transform::new(start, rotation)),
                    additional_mass: Some(MassProperties {
                        mass,
                        local_com: Vec3::ZERO,
                        inertia: Mat3::IDENTITY * (mass / 6.0),
                    }),
                    ..Default::default()
                })
                .unwrap();
            let material = ContactMaterial {
                stiffness,
                damping: 150.0 * mass,
                friction: Some(FrictionCoefficients {
                    mu_static: 0.6,
                    mu_dynamic: 0.2,
                }),
                ..Default::default()
            };
            for x in [-0.4, 0.4] {
                for z in [-0.4, 0.4] {
                    let mut pad = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.1 });
                    pad.parent = Some(body);
                    pad.mass = ColliderMass::Mass(0.0);
                    pad.pose = Transform::from_translation(Vec3::new(x, -0.4, z));
                    pad.contact_material = material;
                    scene.insert_collider(pad).unwrap();
                }
            }
            let mut ground = ColliderDesc::new(ColliderGeometry::Cuboid {
                half_extents: Vec3::new(2.0, 0.1, 2.0),
            });
            ground.pose = Transform::new(offset - normal * 0.1, rotation);
            ground.contact_material = material;
            scene.insert_collider(ground).unwrap();
            cases.push(Case {
                body,
                normal,
                start,
                settled: start,
                peak: 0.0,
                creep: 0.0,
            });
        }
    }
    (scene, cases)
}

fn certify(mut make_world: impl FnMut(RigidScene) -> RigidWorld) {
    let rates = match std::env::var("PRIMITIVE_FRICTION_HZ") {
        Err(std::env::VarError::NotPresent) => vec![60, 120, 240],
        Ok(value) => match value.parse::<usize>().unwrap() {
            hz @ (60 | 120 | 240) => vec![hz],
            _ => panic!("PRIMITIVE_FRICTION_HZ must be 60, 120 or 240"),
        },
        Err(error) => panic!("invalid PRIMITIVE_FRICTION_HZ: {error}"),
    };
    println!("static hold rates={rates:?} duration=10s substeps=8 cases_per_rate=4");
    for hz in rates {
        let (scene, mut cases) = fixture();
        let mut world = make_world(scene);
        world.solver.substeps = 8;
        let dt = 1.0 / hz as f64;
        for frame in 1..=10 * hz {
            world
                .step(dt)
                .unwrap_or_else(|error| panic!("hz={hz} frame={frame}: {error}"));
            for (index, case) in cases.iter_mut().enumerate() {
                let pose = world.scene.body_pose(case.body).unwrap();
                let velocity = world.scene.body_velocity(case.body).unwrap();
                assert!(pose.position.is_finite() && pose.rotation.is_finite());
                assert!(velocity.linear.is_finite() && velocity.angular.is_finite());
                let position = pose.position;
                let tangent = |v: Vec3| v - case.normal * v.dot(case.normal);
                case.peak = case.peak.max(tangent(position - case.start).length());
                if frame == hz {
                    case.settled = position;
                }
                if frame > hz {
                    case.creep = case.creep.max(tangent(position - case.settled).length());
                    assert!(
                        tangent(velocity.linear).length() < 1e-4,
                        "hz={hz} frame={frame} case={index} velocity={velocity:?}"
                    );
                }
                assert!(
                    case.peak <= 1e-3 && case.creep <= 1e-4,
                    "hz={hz} frame={frame} case={index} peak={} creep={}",
                    case.peak,
                    case.creep
                );
                assert!(!world.is_sleeping(case.body));
            }
            assert!(world.quarantined_bodies().is_empty());
            assert!(world.joint_break_events().is_empty());
            assert_eq!(world.completed_steps(), frame as u64);
            if frame > hz {
                let pairs = world.contact_pairs();
                assert_eq!(pairs.len(), cases.len() * 4);
                for pair in pairs {
                    assert!(!pair.points.is_empty());
                    for point in pair.points {
                        assert!(point.normal.is_finite() && point.force_on_b.is_finite());
                        assert!(point.normal_force.is_finite() && point.normal_force >= 0.0);
                        assert!(point.tangent_impulse.is_finite());
                        assert_eq!(point.friction_regime, FrictionRegime::Sticking);
                        assert!((point.applied_friction - 0.6).abs() < 1e-6);
                    }
                }
            }
            if frame % hz == 0 {
                println!(
                    "static hold progress: hz={hz} elapsed={}s peak={:e} post_settle_creep={:e}",
                    frame / hz,
                    cases.iter().map(|case| case.peak).fold(0.0_f64, f64::max),
                    cases.iter().map(|case| case.creep).fold(0.0_f64, f64::max)
                );
            }
        }
        println!(
            "static hold duration=10s hz={hz} substeps=8 cases={} peak={} post_settle_creep={}",
            cases.len(),
            cases.iter().map(|case| case.peak).fold(0.0_f64, f64::max),
            cases.iter().map(|case| case.creep).fold(0.0_f64, f64::max)
        );
    }
}

#[test]
fn loaded_static_friction_holds_ten_seconds_at_three_step_sizes() {
    certify(|scene| RigidWorld::from_scene(scene).unwrap());
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU through make test-gpu"]
fn gpu_loaded_static_friction_holds_ten_seconds_at_three_step_sizes() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "static hold requires a hardware GPU");
    println!("adapter={adapter:?}");
    certify(|scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap());
}
