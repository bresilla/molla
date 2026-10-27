use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::JointAxis;
use molla_sim::runtime::{
    BodyDesc, BodyHandle, ColliderDesc, ColliderGeometry, JointDesc, RigidScene,
};
use molla_sim::{
    BodyParams, ColliderMass, ContactMaterial, FrictionCoefficients, JointType, MassProperties,
};
use molla_solvers::rigid_world::RigidWorld;

fn rotation(angles: Vec3) -> Quat {
    Quat::from_rotation_x(angles.x)
        * Quat::from_rotation_y(angles.y)
        * Quat::from_rotation_z(angles.z)
}

fn fixture(axis: usize) -> (RigidScene, BodyHandle, Quat) {
    fixture_with_angles(axis, Vec3::new(0.4, -0.3, 0.2))
}

fn fixture_with_angles(axis: usize, angles: Vec3) -> (RigidScene, BodyHandle, Quat) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let (body, parent) = append_fixture(&mut scene, axis, angles);
    (scene, body, parent)
}

fn append_fixture(scene: &mut RigidScene, axis: usize, angles: Vec3) -> (BodyHandle, Quat) {
    append_fixture_at(scene, axis, angles, Vec3::new(2., 3., -1.))
}

fn append_fixture_at(
    scene: &mut RigidScene,
    axis: usize,
    angles: Vec3,
    position: Vec3,
) -> (BodyHandle, Quat) {
    let parent = Transform::new(position, Quat::from_rotation_y(0.2));
    let child = Transform::from_rotation(Quat::from_rotation_x(0.1));
    let pose = parent
        .compose(&Transform::from_rotation(rotation(angles)))
        .compose(&child.inverse());
    let body = scene
        .insert_body(BodyDesc {
            params: BodyParams {
                pose,
                ..Default::default()
            },
            additional_mass: Some(MassProperties {
                mass: 1.,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap();
    let joint = scene
        .insert_joint(JointDesc {
            parent: None,
            child: body,
            kind: JointType::D6,
            axis: Vec3::X,
            frame_parent: parent,
            frame_child: child,
        })
        .unwrap();
    scene
        .set_joint_limits(
            joint,
            [JointAxis::AngX, JointAxis::AngY, JointAxis::AngZ][axis],
            Some([-0.1, 0.1]),
        )
        .unwrap();
    scene.set_body_pose(body, pose).unwrap();
    (body, parent.rotation)
}

fn verify(world: &mut RigidWorld, body: BodyHandle, parent: Quat, axis: usize, tolerance: f64) {
    let pose = verify_bounds(world, body, parent, axis, tolerance);
    verify_free_response(world, body, parent, axis, tolerance, pose);
}

fn verify_bounds(
    world: &RigidWorld,
    body: BodyHandle,
    parent: Quat,
    axis: usize,
    tolerance: f64,
) -> Transform {
    let pose = world.scene.body_pose(body).unwrap();
    let relative = parent.conjugate() * pose.rotation * Quat::from_rotation_x(0.1);
    let (x, y, z) = relative.to_euler(glam::EulerRot::XYZ);
    let angles = [x, y, z];
    let violation = (angles[axis].abs() - 0.1).max(0.);
    println!("bounded_xyz axis={axis} angles={angles:?} violation={violation:e}");
    assert!(
        violation <= tolerance,
        "axis={axis} violation={violation:e}"
    );
    assert!((pose.position - Vec3::new(2., 3., -1.)).length() <= tolerance);
    assert!(world.quarantined_bodies().is_empty());
    pose
}

fn verify_free_response(
    world: &mut RigidWorld,
    body: BodyHandle,
    parent: Quat,
    axis: usize,
    tolerance: f64,
    pose: Transform,
) {
    let relative = parent.conjugate() * pose.rotation * Quat::from_rotation_x(0.1);
    let (x, y, _) = relative.to_euler(glam::EulerRot::XYZ);
    let normal = match axis {
        0 => Vec3::new(y.cos(), x.sin() * y.sin(), -x.cos() * y.sin()),
        1 => Vec3::new(0., x.cos(), x.sin()),
        _ => Vec3::new(0., -x.sin(), x.cos()),
    }
    .normalize();
    let reference = [Vec3::X, Vec3::Y, Vec3::Z]
        .into_iter()
        .min_by(|a, b| normal.dot(*a).abs().total_cmp(&normal.dot(*b).abs()))
        .unwrap();
    let tangent = normal.cross(reference).normalize();
    let translation_rate = Vec3::new(0.02, -0.03, 0.01);
    for direction in [tangent, normal.cross(tangent).normalize()] {
        world.scene.set_body_pose(body, pose).unwrap();
        world
            .scene
            .set_body_velocity(
                body,
                SpatialVector::new(translation_rate, parent * direction * 0.05),
            )
            .unwrap();
        world.step(1e-6).unwrap();
        let actual = world.scene.body_velocity(body).unwrap();
        let rate = actual.angular.dot(parent * direction);
        assert!(
            (rate - 0.05).abs() <= tolerance,
            "free-axis rate={rate:e} axis={axis}"
        );
        assert!(rate > 0.045, "free angular motion was frozen");
        let linear_error = (actual.linear - translation_rate).length();
        println!("free_response axis={axis} rate={rate:e} linear_error={linear_error:e}");
        assert!(linear_error <= tolerance);
        assert!(actual.linear.length() > 0.03, "free translation was frozen");
        verify_bounds(world, body, parent, axis, tolerance);
        assert!(world.quarantined_bodies().is_empty());
    }
}

#[test]
fn d6_near_gimbal_projection_satisfies_bounds_and_keeps_free_response() {
    for sign in [-1., 1.] {
        for axis in 0..3 {
            let angles = Vec3::new(0.4, sign * (std::f64::consts::FRAC_PI_2 - 0.003), 0.2);
            let (scene, body, expected) = fixture_with_angles(axis, angles);
            let mut world = RigidWorld::from_scene(scene).unwrap();
            world.solver.substeps = 1;
            world.step(0.001).unwrap();
            verify(&mut world, body, expected, axis, 1e-7);
        }
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; run via make test-primitives-multidof-feedback-gpu"]
fn gpu_d6_projection_satisfies_xyz_bounds_and_keeps_free_response() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine
        .adapter_info()
        .expect("requires GPU adapter information");
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?}");
    let (scene, cases) = batch_fixture(false);
    let world = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    verify_batch(world, cases);
}

type BoundCase = (BodyHandle, Quat, usize, f64);

fn batch_fixture(common_rows: bool) -> (RigidScene, Vec<BoundCase>) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    if common_rows {
        let pose = Transform::from_translation(Vec3::X * 100.);
        let body = scene
            .insert_body(BodyDesc {
                params: BodyParams::new().with_pose(pose),
                additional_mass: Some(MassProperties {
                    mass: 1.,
                    local_com: Vec3::ZERO,
                    inertia: Mat3::IDENTITY,
                }),
                ..Default::default()
            })
            .unwrap();
        scene
            .insert_rigid_loop_joint(JointDesc {
                parent: None,
                child: body,
                kind: JointType::Fixed,
                axis: Vec3::X,
                frame_parent: pose,
                frame_child: Transform::IDENTITY,
            })
            .unwrap();
        assert_eq!(scene.rigid_loop_joints().count(), 1);
    }
    let mut cases = Vec::new();
    for axis in 0..3 {
        let (body, parent) = append_fixture(&mut scene, axis, Vec3::new(0.4, -0.3, 0.2));
        cases.push((body, parent, axis, 1e-5));
    }
    for sign in [-1., 1.] {
        for axis in 0..3 {
            let angles = Vec3::new(0.4, sign * (std::f64::consts::FRAC_PI_2 - 0.003), 0.2);
            let (body, parent) = append_fixture(&mut scene, axis, angles);
            cases.push((body, parent, axis, 1e-3));
        }
    }
    (scene, cases)
}

fn verify_batch(world: RigidWorld, cases: Vec<BoundCase>) -> RigidWorld {
    verify_batch_with_contacts(world, cases, &[])
}

fn verify_batch_with_contacts(
    mut world: RigidWorld,
    cases: Vec<BoundCase>,
    direct: &[BoundCase],
) -> RigidWorld {
    assert_eq!(cases.len(), 9);
    world.solver.substeps = 1;
    world.step(0.001).unwrap();
    let pairs = world.contact_pairs();
    for &(body, parent, axis, tolerance) in direct {
        let pose = world.scene.body_pose(body).unwrap();
        let relative = parent.conjugate() * pose.rotation * Quat::from_rotation_x(0.1);
        let (x, y, z) = relative.to_euler(glam::EulerRot::XYZ);
        let violation = ([x, y, z][axis].abs() - 0.1).max(0.);
        assert!(
            violation <= tolerance,
            "direct_contact axis={axis} violation={violation:e}"
        );
        let pair = pairs
            .iter()
            .find(|pair| pair.bodies.contains(&Some(body)))
            .expect("direct D6 contact must be published at the solved stamp");
        let mut torque = Vec3::ZERO;
        let mut normal_force = 0.;
        for point in &pair.points {
            let sign = if pair.bodies[1] == Some(body) {
                1.
            } else {
                -1.
            };
            torque += (point.point - pose.position).cross(point.force_on_b * sign);
            normal_force += point.normal_force;
        }
        assert!(
            normal_force > 0. && torque.length() > 1e-5,
            "direct_contact axis={axis} force={normal_force} torque={torque:?}"
        );
        assert!(world.quarantined_bodies().is_empty());
        println!(
            "direct_contact axis={axis} angles={:?} violation={violation:e} normal_force={normal_force} torque={torque:?}",
            [x, y, z]
        );
    }
    let poses: Vec<_> = cases
        .iter()
        .map(|&(body, parent, axis, tolerance)| {
            verify_bounds(&world, body, parent, axis, tolerance)
        })
        .collect();
    for &(body, _, _, _) in &cases {
        world
            .scene
            .set_body_velocity(body, SpatialVector::ZERO)
            .unwrap();
    }
    let mut previous = None;
    for ((body, parent, axis, tolerance), pose) in cases.into_iter().zip(poses) {
        if let Some(body) = previous {
            world
                .scene
                .set_body_velocity(body, SpatialVector::ZERO)
                .unwrap();
        }
        verify_free_response(&mut world, body, parent, axis, tolerance, pose);
        previous = Some(body);
    }
    assert_eq!(world.completed_steps(), 19);
    world
}

fn contact_batch_fixture() -> (RigidScene, Vec<BoundCase>, Vec<BoundCase>) {
    let (mut scene, cases) = batch_fixture(true);
    let dummy = scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_position(Vec3::Y * 0.49998),
            additional_mass: Some(MassProperties {
                mass: 1.,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap();
    let material = ContactMaterial {
        stiffness: 1e5,
        damping: 0.,
        friction: Some(FrictionCoefficients {
            mu_static: 0.5,
            mu_dynamic: 0.3,
        }),
        ..Default::default()
    };
    let mut ball = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
    ball.parent = Some(dummy);
    ball.mass = ColliderMass::Mass(0.);
    ball.contact_material = material;
    scene.insert_collider(ball).unwrap();
    let mut ground = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(5., 0.1, 5.),
    });
    ground.pose = Transform::from_translation(-Vec3::Y * 0.1);
    ground.contact_material = material;
    scene.insert_collider(ground).unwrap();
    let mut direct = Vec::new();
    for axis in 0..3 {
        let position = Vec3::new(10. + 8. * axis as f64, 3., -1.);
        let (body, parent) =
            append_fixture_at(&mut scene, axis, Vec3::new(0.4, -0.3, 0.2), position);
        let pose = scene.body_pose(body).unwrap();
        let offset = Vec3::new(0.3, -0.5, -0.2);
        let mut ball = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.2 });
        ball.parent = Some(body);
        ball.mass = ColliderMass::Mass(0.);
        ball.contact_material = material;
        ball.pose = Transform::from_translation(pose.rotation.conjugate() * offset);
        scene.insert_collider(ball).unwrap();
        let mut ground = ColliderDesc::new(ColliderGeometry::Cuboid {
            half_extents: Vec3::new(2., 0.1, 2.),
        });
        ground.pose = Transform::from_translation(position + Vec3::Y * (-0.79998));
        ground.contact_material = material;
        scene.insert_collider(ground).unwrap();
        direct.push((body, parent, axis, 1e-5));
    }
    (scene, cases, direct)
}

#[test]
fn d6_contact_rows_satisfy_xyz_bounds_and_keep_free_response() {
    let (scene, mut cases, mut direct) = contact_batch_fixture();
    for (_, _, _, tolerance) in cases.iter_mut().chain(&mut direct) {
        *tolerance = 1e-7;
    }
    let world = verify_batch_with_contacts(RigidWorld::from_scene(scene).unwrap(), cases, &direct);
    assert!(
        world
            .contact_pairs()
            .iter()
            .any(|pair| pair.points.iter().any(|point| point.normal_force > 0.))
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; run via make test-primitives-multidof-feedback-gpu"]
fn gpu_d6_contact_rows_satisfy_xyz_bounds_and_keep_free_response() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine
        .adapter_info()
        .expect("requires GPU adapter information");
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?} route=contact_rows");
    let (scene, cases, direct) = contact_batch_fixture();
    let world = verify_batch_with_contacts(
        RigidWorld::from_scene_gpu(scene, engine).unwrap(),
        cases,
        &direct,
    );
    assert!(
        world
            .contact_pairs()
            .iter()
            .any(|pair| pair.points.iter().any(|point| point.normal_force > 0.))
    );
}

#[test]
fn d6_common_rows_satisfy_xyz_bounds_and_keep_free_response() {
    let (scene, mut cases) = batch_fixture(true);
    for (_, _, _, tolerance) in &mut cases {
        *tolerance = 1e-7;
    }
    println!("route=common_rows external_fixed_loops=1");
    verify_batch(RigidWorld::from_scene(scene).unwrap(), cases);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; run via make test-primitives-multidof-feedback-gpu"]
fn gpu_d6_common_rows_satisfy_xyz_bounds_and_keep_free_response() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine
        .adapter_info()
        .expect("requires GPU adapter information");
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?} route=common_rows external_fixed_loops=1");
    let (scene, cases) = batch_fixture(true);
    verify_batch(RigidWorld::from_scene_gpu(scene, engine).unwrap(), cases);
}

#[test]
fn d6_angular_projection_satisfies_xyz_bounds_and_keeps_free_response() {
    for axis in 0..3 {
        let (scene, body, expected) = fixture(axis);
        let mut world = RigidWorld::from_scene(scene).unwrap();
        world.solver.substeps = 1;
        world.step(0.001).unwrap();
        verify(&mut world, body, expected, axis, 1e-7);
    }
}
