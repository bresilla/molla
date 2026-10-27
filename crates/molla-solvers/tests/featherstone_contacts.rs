use molla_core::BodyId;
use molla_math::{SpatialVector, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, ColliderDesc, ColliderGeometry, RigidScene};
use molla_sim::{
    BodyParams, CoefficientCombineRule, ContactMaterial, Contacts, ModelBuilder, ShapeFlags,
};
use molla_solvers::penalty_contact::eval_penalty_contacts;
use molla_solvers::rigid_world::RigidWorld;
use molla_solvers::{FeatherstoneSolver, Solver};

fn material(ke: f64, kd: f64) -> ContactMaterial {
    ContactMaterial {
        stiffness: ke,
        damping: kd,
        friction_stiffness: 1000.0,
        friction_smoothing: 0.01,
        friction_combine: CoefficientCombineRule::Average,
        friction: None,
    }
}

#[test]
fn penalty_normal_force_and_com_torques_match_analytic_values() {
    let mut builder = ModelBuilder::new();
    let a = builder.add_body(BodyParams::new().with_com(Vec3::Y * 0.5));
    let b = builder.add_body(BodyParams::new().with_position(Vec3::X));
    let sa = builder
        .add_shape_sphere(a, 0.5, Transform::IDENTITY)
        .unwrap();
    let sb = builder
        .add_shape_sphere(b, 0.5, Transform::IDENTITY)
        .unwrap();
    builder
        .set_shape_contact_material(sa, material(2e4, 100.0))
        .unwrap();
    builder
        .set_shape_contact_material(sb, material(4e4, 300.0))
        .unwrap();
    let model = builder.build().unwrap();
    let mut state = model.state().unwrap();
    state.body_qd.host_mut().unwrap()[0].linear = Vec3::X;
    let mut contacts = Contacts::with_capacity(1);
    contacts.count = 1;
    contacts.a_index.host_mut().unwrap()[0] = a.0;
    contacts.b_index.host_mut().unwrap()[0] = b.0;
    contacts.a_shape.host_mut().unwrap()[0] = sa.0;
    contacts.b_shape.host_mut().unwrap()[0] = sb.0;
    contacts.point.host_mut().unwrap()[0] = Vec3::Y * 2.0;
    contacts.normal.host_mut().unwrap()[0] = Vec3::X;
    contacts.depth.host_mut().unwrap()[0] = 0.01;
    let mut forces = vec![SpatialVector::ZERO; 2];
    let readings = eval_penalty_contacts(&model, &state, &contacts, &mut forces).unwrap();
    assert_eq!(readings[0].normal_force, 500.0);
    assert_eq!(forces[0].linear, -Vec3::X * 500.0);
    assert_eq!(forces[1].linear, Vec3::X * 500.0);
    assert_eq!(forces[0].angular, Vec3::Z * 750.0);
    let total_world_torque = forces[0].angular
        + (Vec3::Y * 0.5).cross(forces[0].linear)
        + forces[1].angular
        + Vec3::X.cross(forces[1].linear);
    assert!(total_world_torque.length() < 1e-12);
    state.body_qd.host_mut().unwrap()[0].linear = -Vec3::X;
    forces.fill(SpatialVector::ZERO);
    let readings = eval_penalty_contacts(&model, &state, &contacts, &mut forces).unwrap();
    assert_eq!(readings[0].normal_force, 300.0);
}

#[test]
fn friction_opposes_slip_is_finite_at_zero_and_never_exceeds_coulomb() {
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(BodyParams::new());
    let model = builder.build().unwrap();
    let mut state = model.state().unwrap();
    let mut contacts = Contacts::with_capacity(1);
    contacts.count = 1;
    contacts.a_index.host_mut().unwrap()[0] = u32::MAX;
    contacts.b_index.host_mut().unwrap()[0] = body.0;
    contacts.normal.host_mut().unwrap()[0] = Vec3::Y;
    contacts.depth.host_mut().unwrap()[0] = 0.001;
    contacts.friction.host_mut().unwrap()[0] = 0.3;
    for speed in [0.0, 1e-12, 1e-6, 0.005, 0.01, 0.1, 10.0, 1000.0] {
        state.body_qd.host_mut().unwrap()[0].linear = Vec3::X * speed;
        let mut forces = vec![SpatialVector::ZERO];
        let reading = eval_penalty_contacts(&model, &state, &contacts, &mut forces)
            .unwrap()
            .remove(0);
        assert!(reading.force_on_b.is_finite());
        assert!(reading.force_on_b.x <= 0.0);
        assert!(reading.force_on_b.x.abs() <= 0.3 * reading.normal_force + 1e-12);
    }
    contacts.depth.host_mut().unwrap()[0] = 0.0;
    let mut forces = vec![SpatialVector::ZERO];
    let reading = eval_penalty_contacts(&model, &state, &contacts, &mut forces)
        .unwrap()
        .remove(0);
    assert!(!reading.acted);
    assert_eq!(forces[0], SpatialVector::ZERO);
    contacts.depth.host_mut().unwrap()[0] = f64::NAN;
    assert!(eval_penalty_contacts(&model, &state, &contacts, &mut forces).is_err());
    assert_eq!(forces[0], SpatialVector::ZERO);
}

fn falling_world(
    height: f64,
    speed: f64,
) -> (
    RigidWorld,
    molla_sim::runtime::BodyHandle,
    molla_sim::runtime::ColliderHandle,
) {
    let mut scene = RigidScene::new().unwrap();
    let mut desc = BodyDesc::default();
    desc.params.pose.position.y = height;
    desc.velocity.linear.y = speed;
    let body = scene.insert_body(desc).unwrap();
    let mut ball = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
    ball.parent = Some(body);
    scene.insert_collider(ball).unwrap();
    let mut ground = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(10.0, 0.5, 10.0),
    });
    ground.pose.position.y = -0.5;
    let ground = scene.insert_collider(ground).unwrap();
    (RigidWorld::from_scene(scene).unwrap(), body, ground)
}

#[test]
fn force_readout_supports_weight_and_does_not_accumulate_in_applied_forces() {
    let (mut world, body, _) = falling_world(1.2, 0.0);
    for _ in 0..600 {
        world.step(1.0 / 120.0).unwrap();
    }
    let i = world.scene.body_index(body).unwrap().index();
    let height = world.scene.state().body_q.host().unwrap()[i].position.y;
    assert!(
        (height - (0.5 - 9.81 / 1e5)).abs() < 1e-5,
        "height={height}"
    );
    let pairs = world.contact_pairs();
    assert_eq!(pairs.len(), 1);
    assert!(pairs[0].points[0].acted);
    assert!((pairs[0].points[0].normal_force - 9.81).abs() < 0.05);
    let impulse: f64 = pairs[0].points.iter().map(|p| p.normal_impulse).sum();
    assert!((impulse - 9.81 / 120.0).abs() < 0.001);
    assert_eq!(
        world.scene.state().body_f.host().unwrap()[i],
        SpatialVector::ZERO
    );
    world.scene.remove_body(body).unwrap();
    assert!(world.contact_pairs().is_empty());
}

#[test]
fn collision_is_refreshed_after_contact_free_first_substep() {
    let (mut world, body, _) = falling_world(0.54, -2.0);
    world.scene.set_gravity(Vec3::ZERO).unwrap();
    world.solver.substeps = 32;
    world.step(1.0 / 30.0).unwrap();
    let i = world.scene.body_index(body).unwrap().index();
    let pose = world.scene.state().body_q.host().unwrap()[i];
    assert!(pose.position.y > 0.49, "height={}", pose.position.y);
    assert!(!world.contact_pairs().is_empty());
}

#[test]
fn filtered_pairs_are_changeable_each_step() {
    let (mut world, body, _) = falling_world(0.49, 0.0);
    world.scene.set_gravity(Vec3::ZERO).unwrap();
    let mut calls = 0;
    world
        .step_filtered(1.0 / 120.0, |a, b| {
            calls += 1;
            a == Some(body) || b == Some(body)
        })
        .unwrap();
    assert!(calls >= 8);
    assert!(world.contact_pairs().is_empty());
    world.step(1.0 / 120.0).unwrap();
    assert!(!world.contact_pairs().is_empty());
}

#[test]
fn runtime_material_sensor_enable_and_group_edits_take_effect() {
    let (mut world, _, ground) = falling_world(0.49, 0.0);
    world.scene.set_gravity(Vec3::ZERO).unwrap();
    let mut desc = world
        .scene
        .colliders()
        .find(|(id, _)| *id == ground)
        .unwrap()
        .1
        .clone();
    desc.friction = 0.2;
    desc.contact_material.friction_combine = CoefficientCombineRule::Min;
    desc.sensor = true;
    world.scene.replace_collider(ground, desc.clone()).unwrap();
    world.step(1.0 / 120.0).unwrap();
    let reading = &world.contact_pairs()[0].points[0];
    assert_eq!(reading.friction, 0.2);
    assert!(!reading.acted);
    assert_eq!(reading.normal_force, 0.0);
    desc.enabled = false;
    world.scene.replace_collider(ground, desc.clone()).unwrap();
    world.step(1.0 / 120.0).unwrap();
    assert!(world.contact_pairs().is_empty());
    desc.enabled = true;
    desc.filter = 0;
    world.scene.replace_collider(ground, desc.clone()).unwrap();
    world.step(1.0 / 120.0).unwrap();
    assert!(world.contact_pairs().is_empty());
    desc.filter = u32::MAX;
    desc.sensor = false;
    world.scene.replace_collider(ground, desc).unwrap();
    world.step(1.0 / 120.0).unwrap();
    assert!(world.contact_pairs()[0].points[0].acted);
}

#[test]
fn supplied_contacts_are_not_accumulated_across_solver_substeps() {
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(molla_core::WorldId(0), Vec3::ZERO);
    let body = builder.add_body(BodyParams::new());
    builder
        .add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
        .unwrap();
    let shape = builder
        .add_shape_sphere(body, 0.5, Transform::IDENTITY)
        .unwrap();
    builder
        .set_shape_contact_material(shape, material(1000.0, 0.0))
        .unwrap();
    let model = builder.build().unwrap();
    let input = model.state().unwrap();
    let mut contacts = Contacts::with_capacity(1);
    contacts.count = 1;
    contacts.a_index.host_mut().unwrap()[0] = u32::MAX;
    contacts.b_index.host_mut().unwrap()[0] = body.0;
    contacts.b_shape.host_mut().unwrap()[0] = shape.0;
    contacts.normal.host_mut().unwrap()[0] = Vec3::Y;
    contacts.depth.host_mut().unwrap()[0] = 0.01;
    for n in [1, 2, 8] {
        let mut output = model.state().unwrap();
        let mut solver = FeatherstoneSolver::new_cpu().with_substeps(n);
        solver
            .step(
                &model,
                &input,
                &model.control(),
                &contacts,
                0.01,
                &mut output,
            )
            .unwrap();
        assert!((output.body_qd.host().unwrap()[0].linear.y - 0.1).abs() < 1e-12);
        assert_eq!(output.body_f.host().unwrap()[0], SpatialVector::ZERO);
    }
}

#[test]
fn deterministic_replay_and_invalid_steps_preserve_scene_state() {
    let run = || {
        let (mut world, _, _) = falling_world(0.8, -1.0);
        for _ in 0..120 {
            world.step(1.0 / 120.0).unwrap();
        }
        let poses = world.scene.state().body_q.host().unwrap().to_vec();
        assert!(world.step(f64::NAN).is_err());
        assert!(world.step(0.0).is_err());
        assert_eq!(world.scene.state().body_q.host().unwrap(), poses);
        poses
    };
    assert_eq!(run(), run());
}

#[test]
fn fixed_body_and_static_world_contacts_do_not_report_solver_action() {
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(BodyParams::new().with_flags(molla_sim::BodyFlags::FIXED));
    let shape = builder
        .add_shape_sphere(body, 0.5, Transform::IDENTITY)
        .unwrap();
    builder
        .set_shape_flags(shape, ShapeFlags::COLLIDES)
        .unwrap();
    let model = builder.build().unwrap();
    let mut contacts = Contacts::with_capacity(1);
    contacts.count = 1;
    contacts.a_index.host_mut().unwrap()[0] = body.0;
    contacts.b_index.host_mut().unwrap()[0] = u32::MAX;
    contacts.normal.host_mut().unwrap()[0] = -Vec3::Y;
    contacts.depth.host_mut().unwrap()[0] = 0.1;
    let mut forces = vec![SpatialVector::ZERO];
    let readings =
        eval_penalty_contacts(&model, &model.state().unwrap(), &contacts, &mut forces).unwrap();
    assert!(!readings[0].acted);
    assert_eq!(forces[0], SpatialVector::ZERO);
}

#[test]
fn disabled_joint_velocity_does_not_create_contact_damping() {
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(BodyParams::new());
    let joint = builder
        .add_joint_prismatic(
            BodyId::NONE,
            body,
            Transform::IDENTITY,
            Transform::IDENTITY,
            Vec3::Y,
        )
        .unwrap();
    builder.set_joint_enabled(joint, false).unwrap();
    let model = builder.build().unwrap();
    let mut input = model.state().unwrap();
    input.joint_qd.host_mut().unwrap()[0] = -100.0;
    let mut contacts = Contacts::with_capacity(1);
    contacts.count = 1;
    contacts.a_index.host_mut().unwrap()[0] = u32::MAX;
    contacts.b_index.host_mut().unwrap()[0] = body.0;
    contacts.normal.host_mut().unwrap()[0] = Vec3::Y;
    contacts.depth.host_mut().unwrap()[0] = 0.01;
    let mut output = model.state().unwrap();
    let mut solver = FeatherstoneSolver::new_cpu();
    solver
        .step(
            &model,
            &input,
            &model.control(),
            &contacts,
            0.001,
            &mut output,
        )
        .unwrap();
    assert_eq!(solver.contact_forces()[0].normal_force, 1000.0);
    assert_eq!(output.body_qd.host().unwrap()[0], SpatialVector::ZERO);
}
