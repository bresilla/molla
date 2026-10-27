use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::JointAxis;
use molla_sim::runtime::{
    BodyDesc, BodyHandle, ColliderDesc, ColliderGeometry, JointBreakLimits, JointDesc, JointHandle,
    RigidScene,
};
use molla_sim::{BodyParams, ColliderMass, FrictionCoefficients, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

fn world(scene: RigidScene, gpu: bool) -> RigidWorld {
    if gpu {
        #[cfg(feature = "gpu")]
        return RigidWorld::from_scene_gpu(
            scene,
            std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap()),
        )
        .unwrap();
        #[cfg(not(feature = "gpu"))]
        panic!("GPU fixture requires the gpu feature");
    }
    RigidWorld::from_scene(scene).unwrap()
}

fn controlled_body(kind: JointType, com: Vec3) -> (RigidScene, BodyHandle, JointHandle) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let body = scene
        .insert_body(BodyDesc {
            additional_mass: Some(MassProperties {
                mass: 2.,
                local_com: com,
                inertia: Mat3::IDENTITY * 4.,
            }),
            ..Default::default()
        })
        .unwrap();
    let joint = scene
        .insert_joint(JointDesc {
            parent: None,
            child: body,
            kind,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    (scene, body, joint)
}

fn set_control(scene: &mut RigidScene, body: BodyHandle, force: SpatialVector) {
    let index = scene.body_index(body).unwrap().index();
    scene.control_mut().body_external_f.host_mut().unwrap()[index] = force;
}

fn near(actual: SpatialVector, expected: SpatialVector, tolerance: f64) {
    assert!(actual.linear.is_finite() && actual.angular.is_finite());
    assert!(
        (actual.linear - expected.linear).length() < tolerance,
        "{actual:?} != {expected:?}"
    );
    assert!(
        (actual.angular - expected.angular).length() < tolerance,
        "{actual:?} != {expected:?}"
    );
}

fn support_and_break(gpu: bool) {
    let (mut scene, body, joint) = controlled_body(JointType::Fixed, Vec3::X * 0.25);
    let persistent = SpatialVector::new(Vec3::X * 2., Vec3::ZERO);
    scene.add_body_wrench(body, persistent).unwrap();
    let mut world = world(scene, gpu);
    world.solver.substeps = 4;
    world.set_reaction_readout(true).unwrap();
    let tolerance = if gpu { 2e-4 } else { 1e-9 };
    for strength in [6., -2., 0., 6.] {
        let control = SpatialVector::new(Vec3::Y * strength, Vec3::Z * (strength * 0.5));
        set_control(&mut world.scene, body, control);
        let expected = SpatialVector::new(
            -persistent.linear - control.linear,
            -control.angular - Vec3::X.cross(control.linear) * 0.25,
        );
        for _ in 0..2 {
            world.step(0.02).unwrap();
            let reaction = world.joint_reaction(joint).unwrap();
            near(reaction.child, expected, tolerance);
            near(reaction.parent, expected.scale(-1.), tolerance);
            let samples = world.joint_reaction_samples().unwrap();
            assert_eq!(samples.len(), 4);
            for sample in samples {
                near(sample.child, expected, tolerance);
            }
            let index = world.scene.body_index(body).unwrap().index();
            near(
                world.scene.state().body_f.host().unwrap()[index],
                persistent,
                1e-12,
            );
            near(
                world.scene.control().body_external_f.host().unwrap()[index],
                control,
                1e-12,
            );
        }
    }
    #[cfg(feature = "gpu")]
    if gpu {
        assert_eq!(world.gpu_diagnostics().unwrap().model_compilations, 2);
    }
    world
        .scene
        .set_joint_break_limits(
            joint,
            JointBreakLimits {
                force: Some(5.),
                torque: None,
            },
        )
        .unwrap();
    world.step(0.02).unwrap();
    assert_eq!(world.joint_break_events().len(), 1);
    assert_eq!(world.joint_break_events()[0].sample.substep, 0);
    near(
        world.scene.body_velocity(body).unwrap(),
        SpatialVector::new(Vec3::new(1., 3., 0.) * 0.015, Vec3::Z * (0.75 * 0.015)),
        tolerance,
    );
    world.step(0.02).unwrap();
    assert_eq!(world.joint_break_events().len(), 1);
}

fn free_acceleration(gpu: bool) {
    for substeps in [1, 4] {
        let (mut scene, body, joint) = controlled_body(JointType::Free, Vec3::ZERO);
        let persistent = SpatialVector::new(Vec3::X * 2., Vec3::ZERO);
        let control = SpatialVector::new(Vec3::Y * 6., Vec3::Y * 8.);
        scene.add_body_wrench(body, persistent).unwrap();
        set_control(&mut scene, body, control);
        let mut world = world(scene, gpu);
        world.solver.substeps = substeps;
        world.set_reaction_readout(true).unwrap();
        let acceleration = Vec3::new(1., 3., 0.);
        for frame in 1..=100 {
            world.step(0.01).unwrap();
            let time = frame as f64 * 0.01;
            near(
                world.scene.body_velocity(body).unwrap(),
                SpatialVector::new(acceleration * time, Vec3::Y * (2. * time)),
                1e-3,
            );
            let integral = time * (time + 0.01 / substeps as f64);
            let pose = world.scene.body_pose(body).unwrap();
            assert!((pose.position - acceleration * (integral * 0.5)).length() < 1e-3);
            let expected = Quat::from_rotation_y(integral);
            let error = 2. * pose.rotation.normalize().dot(expected).abs().min(1.).acos();
            assert!(
                error < 1e-3,
                "frame={frame} substeps={substeps} angle={error}"
            );
            near(
                world.joint_reaction(joint).unwrap().child,
                SpatialVector::ZERO,
                0.02,
            );
            let index = world.scene.body_index(body).unwrap().index();
            near(
                world.scene.state().body_f.host().unwrap()[index],
                persistent,
                1e-12,
            );
            near(
                world.scene.control().body_external_f.host().unwrap()[index],
                control,
                1e-12,
            );
        }
    }
}

fn failed_total_wrench_does_not_publish_or_poison_recovery(gpu: bool) {
    let (scene, body, joint) = controlled_body(JointType::Fixed, Vec3::ZERO);
    let mut world = world(scene, gpu);
    world.solver.substeps = 4;
    world.set_reaction_readout(true).unwrap();
    world.step(0.01).unwrap();
    let pose = world.scene.body_pose(body).unwrap();
    let velocity = world.scene.body_velocity(body).unwrap();
    let huge = SpatialVector::new(Vec3::new(f64::MAX, 0., 0.), Vec3::ZERO);
    world.scene.add_body_wrench(body, huge).unwrap();
    set_control(&mut world.scene, body, huge);
    assert!(world.step(0.01).is_err());
    assert_eq!(world.scene.body_pose(body).unwrap(), pose);
    assert_eq!(world.scene.body_velocity(body).unwrap(), velocity);
    assert!(world.joint_reaction(joint).is_none());
    assert!(world.joint_break_events().is_empty());
    world.scene.reset_body_forces(body).unwrap();
    let control = SpatialVector::new(Vec3::X * 4., Vec3::ZERO);
    set_control(&mut world.scene, body, control);
    world.step(0.01).unwrap();
    near(
        world.joint_reaction(joint).unwrap().child,
        control.scale(-1.),
        1e-5,
    );
    assert!(world.scene.body(body).unwrap().enabled);
}

fn contact_holding_and_sliding(gpu: bool) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(-Vec3::Y * 10.).unwrap();
    let pose = Transform::from_translation(Vec3::Y * 0.4999);
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
    let joint = scene
        .insert_joint(JointDesc {
            parent: None,
            child: body,
            kind: JointType::D6,
            axis: Vec3::X,
            frame_parent: pose,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    for axis in [
        JointAxis::LinZ,
        JointAxis::AngX,
        JointAxis::AngY,
        JointAxis::AngZ,
    ] {
        scene.set_joint_limits(joint, axis, Some([0., 0.])).unwrap();
    }
    let mut ball = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
    ball.parent = Some(body);
    ball.mass = ColliderMass::Mass(0.);
    ball.contact_material.stiffness = 1e5;
    ball.contact_material.damping = 300.;
    ball.contact_material.friction = Some(FrictionCoefficients {
        mu_static: 0.8,
        mu_dynamic: 0.4,
    });
    scene.insert_collider(ball.clone()).unwrap();
    let mut ground = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(20., 0.1, 2.),
    });
    ground.pose = Transform::from_translation(-Vec3::Y * 0.1);
    ground.contact_material = ball.contact_material;
    scene.insert_collider(ground).unwrap();
    set_control(
        &mut scene,
        body,
        SpatialVector::new(Vec3::X * 4., Vec3::ZERO),
    );
    let mut world = world(scene, gpu);
    world.solver.substeps = 4;
    world.set_reaction_readout(true).unwrap();
    for frame in 0..2000 {
        world.step(0.005).unwrap();
        let x = world.scene.body_pose(body).unwrap().position.x;
        assert!(x.abs() < 1e-3, "static frame={frame} x={x}");
        assert!(
            world
                .contact_pairs()
                .iter()
                .any(|pair| pair.points.iter().any(|p| p.normal_impulse > 0.))
        );
    }
    set_control(
        &mut world.scene,
        body,
        SpatialVector::new(Vec3::X * 12., Vec3::ZERO),
    );
    let mut initial = 0.;
    for frame in 0..200 {
        if frame == 100 {
            initial = world.scene.body_velocity(body).unwrap().linear.x;
        }
        world.step(0.005).unwrap();
    }
    let acceleration = (world.scene.body_velocity(body).unwrap().linear.x - initial) / 0.5;
    assert!(
        (acceleration - 8.).abs() < 8. * 0.02,
        "sliding acceleration={acceleration}"
    );
    #[cfg(feature = "gpu")]
    if gpu {
        assert_eq!(world.gpu_diagnostics().unwrap().model_compilations, 2);
    }
}

#[test]
fn cpu_control_wrenches_support_break_and_persist_once() {
    support_and_break(false);
}
#[test]
fn cpu_control_wrenches_accelerate_free_bodies_once() {
    free_acceleration(false);
}
#[test]
fn cpu_control_wrenches_hold_then_slide_contacts() {
    contact_holding_and_sliding(false);
}

#[test]
fn cpu_control_wrench_overflow_rolls_back_and_recovers() {
    failed_total_wrench_does_not_publish_or_poison_recovery(false);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU through make test-gpu"]
fn gpu_control_wrenches_support_break_and_persist_once() {
    support_and_break(true);
}
#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU through make test-gpu"]
fn gpu_control_wrenches_accelerate_free_bodies_once() {
    free_acceleration(true);
}
#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU through make test-gpu"]
fn gpu_control_wrenches_hold_then_slide_contacts() {
    contact_holding_and_sliding(true);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU through make test-gpu"]
fn gpu_control_wrench_overflow_rolls_back_and_recovers() {
    failed_total_wrench_does_not_publish_or_poison_recovery(true);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU through make test-gpu"]
fn gpu_optional_empty_body_control_has_valid_zero_dof_bindings() {
    use molla_solvers::{FeatherstoneSolver, Solver};
    use molla_storage::{Buffer, BufferUsage};
    let (mut scene, _, _) = controlled_body(JointType::Fixed, Vec3::ZERO);
    scene.control_mut().body_external_f = Buffer::from_vec(Vec::new(), BufferUsage::Storage);
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    scene.upload_model_to_device(device, queue).unwrap();
    scene.upload_controls_to_device(device, queue).unwrap();
    let mut input = scene.state().clone_host().unwrap();
    let mut output = scene.state().clone_host().unwrap();
    input.upload_to_device(device, queue).unwrap();
    output.upload_to_device(device, queue).unwrap();
    FeatherstoneSolver::new_gpu(&engine, scene.model())
        .unwrap()
        .step(
            scene.model(),
            &input,
            scene.control(),
            &molla_sim::Contacts::with_capacity(0),
            0.01,
            &mut output,
        )
        .unwrap();
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(30)),
        })
        .unwrap();
    engine.check_device().unwrap();
}
