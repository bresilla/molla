use molla_math::{SpatialVector, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, BodyKind, JointDesc, RigidScene};
use molla_sim::{
    BodyFlags, BodyParams, Contacts, JointType, ModelBuilder, eval_body_velocities, eval_fk,
};
use molla_solvers::{FeatherstoneSolver, SemiImplicitSolver, Solver, XpbdSolver};

#[path = "support/world_isolation.rs"]
mod world_isolation;

#[test]
fn independently_created_worlds_isolate_gravity_state_and_mutations() {
    world_isolation::verify(
        |scene| molla_solvers::rigid_world::RigidWorld::from_scene(scene).unwrap(),
        1e-10,
    );
}

fn step(scene: &mut RigidScene, solver: &mut FeatherstoneSolver) {
    let mut next = scene.model().state().unwrap();
    solver
        .step(
            scene.model(),
            scene.state(),
            scene.control(),
            &Contacts::with_capacity(0),
            1.0 / 120.0,
            &mut next,
        )
        .unwrap();
    *scene.state_mut() = next;
}

#[test]
fn fixed_body_stays_at_current_pose_and_can_resume_dynamics() {
    let mut scene = RigidScene::new().unwrap();
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    let mut solver = FeatherstoneSolver::new_cpu().with_rigorous_d_solve(true);
    step(&mut scene, &mut solver);
    assert!(scene.state().body_q.host().unwrap()[0].position.y < 0.0);
    assert!(scene.state().body_qd.host().unwrap()[0].linear.y < 0.0);
    let pose = scene.state().body_q.host().unwrap()[0];
    scene.set_body_kind(body, BodyKind::Fixed).unwrap();
    scene.state_mut().body_f.host_mut().unwrap()[0] =
        SpatialVector::new(Vec3::splat(1e6), Vec3::splat(1e6));
    for _ in 0..120 {
        step(&mut scene, &mut solver);
    }
    assert_eq!(scene.state().body_q.host().unwrap()[0], pose);
    assert_eq!(
        scene.state().body_qd.host().unwrap()[0],
        SpatialVector::ZERO
    );
    scene.state_mut().clear_forces().unwrap();
    scene.set_body_kind(body, BodyKind::Dynamic).unwrap();
    step(&mut scene, &mut solver);
    assert!(scene.state().body_q.host().unwrap()[0].position.y < pose.position.y);
}

#[test]
fn detach_retains_world_pose_and_velocity_after_stepping() {
    let mut scene = RigidScene::new().unwrap();
    let root = scene
        .insert_body(BodyDesc {
            kind: BodyKind::Fixed,
            ..BodyDesc::default()
        })
        .unwrap();
    let mut child = BodyDesc::default();
    child.params.pose.position = Vec3::new(1.0, -1.0, 0.0);
    let body = scene.insert_body(child).unwrap();
    let joint = scene
        .insert_joint(JointDesc {
            parent: Some(root),
            child: body,
            kind: JointType::Revolute,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::from_translation(Vec3::new(-1.0, 1.0, 0.0)),
            axis: Vec3::Z,
        })
        .unwrap();
    let mut solver = FeatherstoneSolver::new_cpu().with_rigorous_d_solve(true);
    for _ in 0..20 {
        step(&mut scene, &mut solver);
    }
    let i = scene.body_index(body).unwrap().index();
    let pose = scene.state().body_q.host().unwrap()[i];
    let velocity = scene.state().body_qd.host().unwrap()[i];
    assert!(velocity.linear.length() > 0.1);
    scene.remove_joint(joint).unwrap();
    let mut state = scene.state().clone_host().unwrap();
    eval_fk(scene.model(), &mut state).unwrap();
    eval_body_velocities(scene.model(), &mut state).unwrap();
    assert!((state.body_q.host().unwrap()[i].position - pose.position).length() < 1e-12);
    assert!((state.body_qd.host().unwrap()[i].linear - velocity.linear).length() < 1e-12);
    assert!((state.body_qd.host().unwrap()[i].angular - velocity.angular).length() < 1e-12);
    step(&mut scene, &mut solver);
    assert!(scene.state().body_q.host().unwrap()[i].position.is_finite());
}

#[test]
fn free_body_dynamics_are_independent_of_world_translation() {
    let simulate = |position: Vec3| {
        let mut scene = RigidScene::new().unwrap();
        scene.set_gravity(Vec3::ZERO).unwrap();
        let mut desc = BodyDesc::default();
        desc.params.pose.position = position;
        desc.velocity = SpatialVector::new(Vec3::new(0.3, -0.2, 0.1), Vec3::new(0.2, 0.5, 0.4));
        scene.insert_body(desc).unwrap();
        let mut solver = FeatherstoneSolver::new_cpu().with_rigorous_d_solve(true);
        for _ in 0..120 {
            step(&mut scene, &mut solver);
        }
        (
            scene.state().body_q.host().unwrap()[0],
            scene.state().body_qd.host().unwrap()[0],
        )
    };
    let offset = Vec3::new(10.0, 3.0, -5.0);
    let (at_origin, velocity) = simulate(Vec3::ZERO);
    let (translated, translated_velocity) = simulate(offset);
    assert!((translated.position - offset - at_origin.position).length() < 1e-9);
    assert!((translated_velocity.linear - velocity.linear).length() < 1e-9);
    assert!((velocity.linear - Vec3::new(0.3, -0.2, 0.1)).length() < 1e-9);
}

#[test]
fn fixed_link_mass_and_forces_reach_the_articulation_root() {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let root = scene.insert_body(BodyDesc::default()).unwrap();
    let mut child = BodyDesc::default();
    child.params.mass = 9.0;
    let body = scene.insert_body(child).unwrap();
    scene
        .insert_joint(JointDesc {
            parent: Some(root),
            child: body,
            kind: JointType::Fixed,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
            axis: Vec3::X,
        })
        .unwrap();
    scene.state_mut().body_f.host_mut().unwrap()[1].linear.x = 20.0;
    let mut solver = FeatherstoneSolver::new_cpu().with_rigorous_d_solve(true);
    step(&mut scene, &mut solver);
    for velocity in scene.state().body_qd.host().unwrap() {
        assert!((velocity.linear.x - 2.0 / 120.0).abs() < 1e-10);
    }
}

#[test]
fn fixed_flags_ignore_input_velocity_and_force_on_cpu_solvers() {
    let mut builder = ModelBuilder::new();
    let pose = Transform::from_translation(Vec3::new(4.0, 3.0, 2.0));
    let body = builder.add_body(
        BodyParams::new()
            .with_pose(pose)
            .with_flags(BodyFlags::FIXED),
    );
    builder
        .add_joint_free(
            molla_core::BodyId::NONE,
            body,
            Transform::IDENTITY,
            Transform::IDENTITY,
        )
        .unwrap();
    let model = builder.build().unwrap();
    let mut input = model.state().unwrap();
    input.body_qd.host_mut().unwrap()[0] =
        SpatialVector::new(Vec3::splat(100.0), Vec3::splat(100.0));
    input.body_f.host_mut().unwrap()[0] =
        SpatialVector::new(Vec3::splat(1000.0), Vec3::splat(1000.0));
    input.joint_qd.host_mut().unwrap().fill(100.0);
    let mut solvers: Vec<Box<dyn Solver>> = vec![
        Box::new(FeatherstoneSolver::new_cpu()),
        Box::new(SemiImplicitSolver::new_cpu()),
        Box::new(XpbdSolver::new_cpu()),
    ];
    for solver in &mut solvers {
        let mut output = model.state().unwrap();
        solver
            .step(
                &model,
                &input,
                &model.control(),
                &Contacts::with_capacity(0),
                0.01,
                &mut output,
            )
            .unwrap();
        assert_eq!(output.body_q.host().unwrap()[0], pose);
        assert_eq!(output.body_qd.host().unwrap()[0], SpatialVector::ZERO);
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a GPU adapter"]
fn fixed_flags_ignore_input_velocity_and_force_on_gpu_integrators() {
    use molla_compute::{Backend, Engine};
    let engine = Engine::new(Backend::AutoGpu).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    let pose = Transform::from_translation(Vec3::new(4.0, 3.0, 2.0));
    builder.add_body(
        BodyParams::new()
            .with_pose(pose)
            .with_flags(BodyFlags::FIXED),
    );
    let mut model = builder.build().unwrap();
    model.upload_to_device(device, queue).unwrap();
    let mut input = model.state().unwrap();
    input.body_qd.host_mut().unwrap()[0] =
        SpatialVector::new(Vec3::splat(100.0), Vec3::splat(100.0));
    input.body_f.host_mut().unwrap()[0] =
        SpatialVector::new(Vec3::splat(1000.0), Vec3::splat(1000.0));
    input.upload_to_device(device, queue).unwrap();
    let mut control = model.control();
    control.upload_to_device(device, queue).unwrap();
    let mut contacts = Contacts::with_capacity(1);
    molla_geometry::CollisionPipeline::new()
        .collide(&model, &input, &mut contacts)
        .unwrap();
    contacts.upload_to_device(device, queue).unwrap();
    let mut solvers: Vec<Box<dyn Solver>> = vec![
        Box::new(SemiImplicitSolver::new_gpu(&engine).unwrap()),
        Box::new(XpbdSolver::new_gpu(&engine, 1).unwrap()),
    ];
    for solver in &mut solvers {
        let mut output = model.state().unwrap();
        output.upload_to_device(device, queue).unwrap();
        solver
            .step(&model, &input, &control, &contacts, 0.01, &mut output)
            .unwrap();
        output.download_from_device(device, queue).unwrap();
        assert_eq!(output.body_q.host().unwrap()[0], pose);
        assert_eq!(output.body_qd.host().unwrap()[0], SpatialVector::ZERO);
    }
}
