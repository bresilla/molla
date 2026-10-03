use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::JointAxis;
use molla_sim::runtime::{
    BodyDesc, BodyHandle, ColliderDesc, ColliderGeometry, JointDesc, RigidScene,
};
use molla_sim::{BodyParams, ColliderMass, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

const DT: f64 = 1. / 240.;
const TOLERANCE: f64 = 1e-3;

fn pole_signs() -> Vec<f64> {
    let signs = match std::env::var("PRIMITIVE_XYZ_POLE") {
        Err(std::env::VarError::NotPresent) => vec![-1., 1.],
        Ok(value) => match value.as_str() {
            "both" => vec![-1., 1.],
            "negative" => vec![-1.],
            "positive" => vec![1.],
            _ => panic!("PRIMITIVE_XYZ_POLE must be both, negative or positive"),
        },
        Err(error) => panic!("invalid PRIMITIVE_XYZ_POLE: {error}"),
    };
    println!("singular_xyz_poles={signs:?}");
    signs
}

fn fixture(sign: f64) -> (RigidScene, BodyHandle) {
    let mut scene = RigidScene::new().unwrap();
    let rotation = Quat::from_euler(
        glam::EulerRot::XYZ,
        0.3,
        sign * std::f64::consts::FRAC_PI_2,
        0.3,
    );
    let body = scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_pose(Transform::new(Vec3::Y, rotation)),
            additional_mass: Some(MassProperties {
                mass: 2.,
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
            frame_parent: Transform::from_translation(Vec3::Y),
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    scene
        .set_joint_limits(joint, JointAxis::AngX, Some([-0.1, 0.1]))
        .unwrap();
    scene
        .set_joint_limits(
            joint,
            JointAxis::AngY,
            Some(if sign > 0. { [1.2, 1.6] } else { [-1.6, -1.2] }),
        )
        .unwrap();
    scene
        .set_joint_limits(joint, JointAxis::AngZ, Some([0.2, 0.4]))
        .unwrap();
    let mut sphere = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.1 });
    sphere.parent = Some(body);
    sphere.mass = ColliderMass::Mass(0.);
    scene.insert_collider(sphere).unwrap();
    let mut floor = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(2., 0.1, 2.),
    });
    floor.pose = Transform::from_translation(-Vec3::Y * 0.1);
    scene.insert_collider(floor).unwrap();
    assert!(phase_violation(scene.body_pose(body).unwrap().rotation, sign) > 0.09);
    (scene, body)
}

fn phase_violation(rotation: Quat, sign: f64) -> f64 {
    let matrix = Mat3::from_quat(rotation.normalize());
    let phase = matrix.y_axis.z.atan2(matrix.y_axis.y);
    let (lower, upper) = if sign > 0. { (0.1, 0.5) } else { (-0.5, -0.1) };
    (lower - phase).max(phase - upper).max(0.)
}

fn advance(world: &mut RigidWorld, body: BodyHandle, sign: f64) {
    world.solver.substeps = 1;
    world.step(DT).unwrap();
    let pose = world.scene.body_pose(body).unwrap();
    let velocity = world.scene.body_velocity(body).unwrap();
    assert!(pose.position.is_finite() && pose.rotation.is_finite());
    assert!(velocity.linear.is_finite() && velocity.angular.is_finite());
    assert!(world.quarantined_bodies().is_empty());
    assert!(world.joint_break_events().is_empty());
    let violation = phase_violation(pose.rotation, sign);
    println!(
        "singular_xyz_frame sign={sign} phase_violation={violation:.9e} velocity={velocity:?}"
    );
    assert!(violation <= TOLERANCE, "phase violation {violation:e}");
    assert!(velocity.angular.x < -1., "inactive phase correction");
}

#[test]
fn cpu_active_phase_interval_at_both_xyz_poles() {
    for sign in pole_signs() {
        let (scene, body) = fixture(sign);
        advance(&mut RigidWorld::from_scene(scene).unwrap(), body, sign);
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU; run make test-primitives-singular-wiring-gpu"]
fn gpu_active_phase_interval_matches_cpu_at_both_xyz_poles() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?}");
    for sign in pole_signs() {
        let (scene, body) = fixture(sign);
        let (cpu_scene, cpu_body) = fixture(sign);
        println!("singular_xyz_prepare_begin sign={sign} pairs=8 contacts_per_island=16");
        let preparation = std::time::Instant::now();
        let mut gpu = RigidWorld::from_scene_gpu_with_contact_capacity(
            scene,
            engine.clone(),
            molla_solvers::rigid_world::GpuContactCapacity {
                pairs: 8,
                contacts_per_island: 16,
            },
        )
        .unwrap();
        println!(
            "singular_xyz_prepare_complete sign={sign} seconds={:.6}",
            preparation.elapsed().as_secs_f64()
        );
        let compilations = gpu.gpu_diagnostics().unwrap().model_compilations;
        let mut cpu = RigidWorld::from_scene(cpu_scene).unwrap();
        advance(&mut cpu, cpu_body, sign);
        println!("singular_xyz_advance_begin sign={sign}");
        let stepping = std::time::Instant::now();
        advance(&mut gpu, body, sign);
        println!(
            "singular_xyz_advance_complete sign={sign} seconds={:.6}",
            stepping.elapsed().as_secs_f64()
        );
        assert_eq!(
            gpu.gpu_diagnostics().unwrap().model_compilations,
            compilations
        );
        assert_eq!(gpu.completed_steps(), 1);
        let a = gpu.scene.body_pose(body).unwrap();
        let b = cpu.scene.body_pose(cpu_body).unwrap();
        let va = gpu.scene.body_velocity(body).unwrap();
        let vb = cpu.scene.body_velocity(cpu_body).unwrap();
        let errors = [
            (a.position - b.position).length(),
            a.rotation.angle_between(b.rotation),
            (va.linear - vb.linear).length(),
            (va.angular - vb.angular).length(),
        ];
        println!("singular_xyz_parity sign={sign} errors={errors:?} tolerance={TOLERANCE}");
        assert!(errors.into_iter().all(|v| v.is_finite() && v <= TOLERANCE));
    }
}

fn crossing_fixture(sign: f64) -> (RigidScene, BodyHandle) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let rotation = Quat::from_rotation_y(sign * (std::f64::consts::FRAC_PI_2 - 0.005));
    let body = scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_pose(Transform::new(Vec3::Y, rotation)),
            additional_mass: Some(MassProperties {
                mass: 2.,
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
            frame_parent: Transform::from_translation(Vec3::Y),
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    for axis in [JointAxis::AngX, JointAxis::AngZ] {
        scene
            .set_joint_limits(joint, axis, Some([-0.1, 0.1]))
            .unwrap();
    }
    scene
        .set_joint_limits(
            joint,
            JointAxis::AngY,
            Some(if sign > 0. { [1.2, 1.9] } else { [-1.9, -1.2] }),
        )
        .unwrap();
    scene
        .set_body_velocity(body, SpatialVector::new(Vec3::ZERO, Vec3::Y * sign))
        .unwrap();
    let mut sphere = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.1 });
    sphere.parent = Some(body);
    sphere.mass = ColliderMass::Mass(0.);
    scene.insert_collider(sphere).unwrap();
    let mut floor = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(2., 0.1, 2.),
    });
    floor.pose = Transform::from_translation(-Vec3::Y * 0.1);
    scene.insert_collider(floor).unwrap();
    (scene, body)
}

fn crossing_frame(world: &mut RigidWorld, body: BodyHandle, sign: f64, frame: usize) {
    world.solver.substeps = 1;
    world.step(0.001).unwrap();
    let actual = world.scene.body_pose(body).unwrap();
    let velocity = world.scene.body_velocity(body).unwrap();
    let initial = std::f64::consts::FRAC_PI_2 - 0.005;
    let previous = (initial + frame as f64 * 0.001).min(1.9);
    let current = (initial + (frame + 1) as f64 * 0.001).min(1.9);
    let expected = Quat::from_rotation_y(sign * current);
    let expected_velocity = Vec3::Y * (sign * (current - previous) / 0.001);
    let rotation = actual.rotation.normalize();
    let dot = rotation.dot(expected).abs().min(1.);
    let angle = 2. * dot.acos();
    assert!(
        angle.is_finite() && angle <= TOLERANCE,
        "frame={frame}, sign={sign}, angle={angle:e}"
    );
    assert!((actual.position - Vec3::Y).length() <= TOLERANCE);
    assert!(velocity.linear.length() <= TOLERANCE);
    assert!(
        (velocity.angular - expected_velocity).length() <= TOLERANCE,
        "frame={frame}, sign={sign}: {velocity:?}"
    );
    assert!(world.quarantined_bodies().is_empty());
    assert!(world.joint_break_events().is_empty());
    assert_eq!(world.completed_steps(), (frame + 1) as u64);
    if (frame + 1).is_multiple_of(100) {
        println!(
            "singular_xyz_crossing sign={sign} elapsed={}s pitch={current} orientation_error={angle:e} angular_rate_error={:e}",
            (frame + 1) as f64 * 0.001,
            (velocity.angular - expected_velocity).length()
        );
    }
}

#[test]
fn cpu_free_d6_motion_crosses_both_xyz_poles_without_false_stops() {
    for sign in pole_signs() {
        let (scene, body) = crossing_fixture(sign);
        let mut world = RigidWorld::from_scene(scene).unwrap();
        for frame in 0..1000 {
            crossing_frame(&mut world, body, sign, frame);
        }
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_free_d6_motion_crosses_both_xyz_poles_without_false_stops() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?}");
    for sign in pole_signs() {
        let (scene, body) = crossing_fixture(sign);
        println!("singular_xyz_crossing_prepare_begin sign={sign} pairs=8 contacts_per_island=16");
        let preparation = std::time::Instant::now();
        let mut world = RigidWorld::from_scene_gpu_with_contact_capacity(
            scene,
            engine.clone(),
            molla_solvers::rigid_world::GpuContactCapacity {
                pairs: 8,
                contacts_per_island: 16,
            },
        )
        .unwrap();
        println!(
            "singular_xyz_crossing_prepare_complete sign={sign} seconds={:.6}",
            preparation.elapsed().as_secs_f64()
        );
        let compilations = world.gpu_diagnostics().unwrap().model_compilations;
        for frame in 0..1000 {
            crossing_frame(&mut world, body, sign, frame);
        }
        assert_eq!(
            world.gpu_diagnostics().unwrap().model_compilations,
            compilations
        );
    }
}
