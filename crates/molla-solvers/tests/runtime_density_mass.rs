use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, ColliderDesc, ColliderGeometry, RigidScene};
use molla_sim::{ColliderMass, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

fn sphere_volume(radius: f64) -> f64 {
    4.0 * std::f64::consts::PI * radius.powi(3) / 3.0
}

fn oracle(masses: [f64; 2], additional: MassProperties) -> MassProperties {
    let centers = [Vec3::new(-0.2, 0.1, 0.3), Vec3::new(0.3, -0.2, 0.1)];
    let radii: [f64; 2] = [0.25, 0.4];
    let mass = masses[0] + masses[1] + additional.mass;
    let local_com =
        (centers[0] * masses[0] + centers[1] * masses[1] + additional.local_com * additional.mass)
            / mass;
    let shift = |center: Vec3, mass: f64| {
        let d = center - local_com;
        (Mat3::IDENTITY * d.length_squared() - Mat3::from_cols(d * d.x, d * d.y, d * d.z)) * mass
    };
    let inertia = Mat3::IDENTITY
        * (0.4 * (masses[0] * radii[0].powi(2) + masses[1] * radii[1].powi(2)))
        + shift(centers[0], masses[0])
        + shift(centers[1], masses[1])
        + additional.inertia
        + shift(additional.local_com, additional.mass);
    MassProperties {
        mass,
        local_com,
        inertia,
    }
}

fn verify(make: impl FnOnce(RigidScene) -> RigidWorld, tolerance: f64) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let mut desc = BodyDesc::default();
    desc.params.pose = Transform::new(
        Vec3::new(0.2, 1.0, 0.3),
        Quat::from_rotation_z(0.4) * Quat::from_rotation_x(-0.3),
    );
    let body = scene.insert_body(desc).unwrap();
    let mut colliders = Vec::new();
    for (radius, mass, center) in [
        (0.25, 2.0, Vec3::new(-0.2, 0.1, 0.3)),
        (0.4, 6.0, Vec3::new(0.3, -0.2, 0.1)),
    ] {
        let mut collider = ColliderDesc::new(ColliderGeometry::Ball { radius });
        collider.parent = Some(body);
        collider.pose.position = center;
        collider.mass = ColliderMass::Density(mass / sphere_volume(radius));
        collider.memberships = 0;
        collider.filter = 0;
        colliders.push(scene.insert_collider(collider).unwrap());
    }
    let unrelated = scene
        .insert_body(BodyDesc {
            velocity: SpatialVector::new(Vec3::X * 0.1, Vec3::ZERO),
            ..Default::default()
        })
        .unwrap();
    let mut world = make(scene);
    world.solver.substeps = 1;
    world
        .set_body_additional_mass(body, MassProperties::ZERO)
        .unwrap();
    let dt = 1.0 / 240.0;
    let force = Vec3::new(12.0, -6.0, 3.0);
    let torque = Vec3::new(0.04, -0.02, 0.07);
    let mut worst_linear: f64 = 0.0;
    let mut worst_angular: f64 = 0.0;
    for phase in 0..3 {
        let pose = world.scene.body_pose(body).unwrap();
        let velocity = world.scene.body_velocity(body).unwrap();
        let origin_velocity = world
            .scene
            .body_velocity_at_point(body, pose.position)
            .unwrap();
        let stamp = world.scene.model_stamp();
        let additional = if phase == 2 {
            MassProperties {
                mass: 2.0,
                local_com: Vec3::new(0.1, 0.2, -0.2),
                inertia: Mat3::from_diagonal(Vec3::new(0.2, 0.3, 0.4)),
            }
        } else {
            MassProperties::ZERO
        };
        if phase == 1 {
            world
                .scene
                .set_collider_mass(
                    colliders[1],
                    ColliderMass::Density(3.0 / sphere_volume(0.4)),
                )
                .unwrap();
            assert!((world.scene.body_mass_properties(body).unwrap().mass - 8.0).abs() < 1e-10);
            world.recompute_body_mass(body).unwrap();
        } else if phase == 2 {
            world.set_body_additional_mass(body, additional).unwrap();
        }
        if phase > 0 {
            assert_ne!(world.scene.model_stamp(), stamp);
        }
        assert_eq!(world.scene.body_pose(body).unwrap(), pose);
        assert_eq!(
            world.scene.body_velocity(body).unwrap().angular,
            velocity.angular
        );
        assert!(
            (world
                .scene
                .body_velocity_at_point(body, pose.position)
                .unwrap()
                - origin_velocity)
                .length()
                < 1e-10
        );
        let expected = oracle([2.0, if phase == 0 { 6.0 } else { 3.0 }], additional);
        let actual = world.scene.body_mass_properties(body).unwrap();
        assert!((actual.mass - expected.mass).abs() < 1e-10);
        assert!((actual.local_com - expected.local_com).length() < 1e-10);
        for (a, b) in actual
            .inertia
            .to_cols_array()
            .into_iter()
            .zip(expected.inertia.to_cols_array())
        {
            assert!((a - b).abs() < 1e-10, "inertia {a} != {b}");
        }
        assert!(expected.inertia.x_axis.y.abs() > 1e-3);
        let before_invalid = world.scene.model_stamp();
        assert!(
            world
                .scene
                .set_collider_mass(colliders[0], ColliderMass::Density(-1.0))
                .is_err()
        );
        assert_eq!(world.scene.model_stamp(), before_invalid);
        assert_eq!(world.scene.body_mass_properties(body).unwrap(), actual);
        world.scene.reset_body_forces(body).unwrap();
        world
            .scene
            .set_body_velocity(body, SpatialVector::ZERO)
            .unwrap();
        world
            .scene
            .add_body_wrench(body, SpatialVector::new(force, torque))
            .unwrap();
        let rotation = Mat3::from_quat(pose.rotation);
        let expected_linear = force * (dt / expected.mass);
        let expected_angular =
            rotation * expected.inertia.inverse() * rotation.transpose() * torque * dt;
        world.step(dt).unwrap();
        let velocity = world.scene.body_velocity(body).unwrap();
        worst_linear = worst_linear.max((velocity.linear - expected_linear).length());
        worst_angular = worst_angular.max((velocity.angular - expected_angular).length());
        assert!(
            worst_linear < tolerance,
            "phase={phase}, linear={worst_linear:e}"
        );
        assert!(
            worst_angular < tolerance,
            "phase={phase}, angular={worst_angular:e}"
        );
        assert!(world.contact_pairs().is_empty());
        assert!(
            (world.scene.body_velocity(unrelated).unwrap().linear - Vec3::X * 0.1).length()
                < tolerance
        );
        assert!(
            (world.scene.body_pose(unrelated).unwrap().position.x - 0.1 * dt * (phase + 1) as f64)
                .abs()
                < tolerance
        );
    }
    eprintln!("density mass edits: max linear={worst_linear:e}, angular={worst_angular:e}");
}

#[test]
fn density_aggregation_and_edits_drive_cpu_force_response() {
    verify(|scene| RigidWorld::from_scene(scene).unwrap(), 2e-8);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU through make test-gpu"]
fn density_aggregation_and_edits_drive_gpu_force_response() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let info = engine.adapter_info().unwrap();
    assert!(!info.is_software, "requires a hardware GPU");
    println!("adapter={info:?}");
    verify(
        |scene| RigidWorld::from_scene_gpu(scene, engine).unwrap(),
        2e-5,
    );
}
