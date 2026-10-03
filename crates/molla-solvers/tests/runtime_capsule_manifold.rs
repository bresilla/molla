use molla_math::{Mat3, Quat, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, BodyHandle, ColliderDesc, ColliderGeometry, RigidScene};
use molla_sim::{BodyParams, ColliderMass, ContactMaterial, FrictionCoefficients, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

struct Case {
    body: BodyHandle,
    pose: Transform,
    axis: Vec3,
    mass: f64,
}

fn fixture() -> (RigidScene, Vec<Case>) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(-Vec3::Y * 10.0).unwrap();
    let mut cases = Vec::new();
    for mass in [1.0, 100.0] {
        for yaw in [0.0, 0.7] {
            let rotation =
                Quat::from_rotation_y(yaw) * Quat::from_rotation_z(std::f64::consts::FRAC_PI_2);
            let center = Vec3::new(8.0 * cases.len() as f64, -3.0, 2.0);
            let stiffness = mass * 1e5;
            let pose = Transform::new(
                center + Vec3::Y * (0.2 - mass * 10.0 / (2.0 * stiffness)),
                rotation,
            );
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
            let material = ContactMaterial {
                stiffness,
                damping: mass * 150.0,
                friction: Some(FrictionCoefficients {
                    mu_static: 0.0,
                    mu_dynamic: 0.0,
                }),
                ..Default::default()
            };
            for parent in [None, Some(body)] {
                let mut shape = ColliderDesc::new(ColliderGeometry::Capsule {
                    half_height: 1.0,
                    radius: 0.1,
                });
                shape.parent = parent;
                shape.mass = ColliderMass::Mass(0.0);
                shape.pose = if parent.is_none() {
                    Transform::new(center, rotation)
                } else {
                    Transform::IDENTITY
                };
                shape.contact_material = material;
                scene.insert_collider(shape).unwrap();
            }
            cases.push(Case {
                body,
                pose,
                axis: rotation * Vec3::Y,
                mass,
            });
        }
    }
    (scene, cases)
}

fn certify(mut make_world: impl FnMut(RigidScene) -> RigidWorld) {
    for hz in [60, 120, 240] {
        let (scene, cases) = fixture();
        let mut world = make_world(scene);
        world.solver.substeps = 8;
        let mut peak_position = 0.0_f64;
        let mut peak_angle = 0.0_f64;
        for frame in 1..=hz {
            world
                .step(1.0 / hz as f64)
                .unwrap_or_else(|error| panic!("hz={hz} frame={frame}: {error}"));
            assert!(world.quarantined_bodies().is_empty());
            let pairs = world.contact_pairs();
            assert_eq!(pairs.len(), cases.len(), "hz={hz} frame={frame}");
            for (index, case) in cases.iter().enumerate() {
                let pose = world.scene.body_pose(case.body).unwrap();
                let velocity = world.scene.body_velocity(case.body).unwrap();
                peak_position = peak_position.max((pose.position - case.pose.position).length());
                peak_angle = peak_angle.max(
                    (case.pose.rotation.conjugate() * pose.rotation)
                        .to_scaled_axis()
                        .length(),
                );
                assert!(
                    peak_position < 1e-3 && peak_angle < 1e-3,
                    "hz={hz} frame={frame} case={index} position={peak_position} angle={peak_angle}"
                );
                assert!(velocity.linear.is_finite() && velocity.angular.is_finite());
                let pair = pairs
                    .iter()
                    .find(|pair| pair.bodies.contains(&Some(case.body)))
                    .unwrap();
                assert!(
                    (2..=32).contains(&pair.points.len()),
                    "hz={hz} frame={frame} case={index} points={}",
                    pair.points.len()
                );
                let mut support = [false; 2];
                let mut force = Vec3::ZERO;
                let mut torque = Vec3::ZERO;
                for point in &pair.points {
                    let along = (point.point - pose.position).dot(case.axis);
                    if along.abs() > 0.99 {
                        support[usize::from(along > 0.0)] = true;
                    }
                    assert!(
                        along.abs() < 1.01,
                        "invalid support witness case={index} along={along}"
                    );
                    assert!((point.sample_dt - 1.0 / (8.0 * hz as f64)).abs() < 1e-12);
                    let sign = if pair.bodies[1] == Some(case.body) {
                        1.0
                    } else {
                        -1.0
                    };
                    let sample = point.force_on_b * sign / 8.0;
                    force += sample;
                    torque += (point.point - pose.position).cross(sample);
                }
                assert_eq!(support, [true, true]);
                if frame > hz / 5 {
                    assert!(
                        (force - Vec3::Y * case.mass * 10.0).length() < 0.01 * case.mass * 10.0,
                        "hz={hz} frame={frame} case={index} force={force:?}"
                    );
                    assert!(
                        torque.length() < 0.01 * case.mass * 10.0,
                        "hz={hz} frame={frame} case={index} torque={torque:?}"
                    );
                }
            }
        }
        eprintln!("capsule span hz={hz} peak_position={peak_position:e} peak_angle={peak_angle:e}");
    }
}

#[test]
fn cpu_capsule_support_span_preserves_loaded_equilibrium() {
    certify(|scene| RigidWorld::from_scene(scene).unwrap());
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn gpu_capsule_support_span_preserves_loaded_equilibrium() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    certify(|scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap());
}
