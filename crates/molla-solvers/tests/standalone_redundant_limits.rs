use molla_core::{BodyId, WorldId};
use molla_math::{Mat3, Quat, SpatialInertia, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::JointAxis;
use molla_sim::runtime::{
    BodyDesc, BodyHandle, DistanceLimits, JointDesc, JointHandle, RigidScene,
};
use molla_sim::{
    BodyParams, Contacts, Control, JointType, MassProperties, Model, ModelBuilder, State,
};
use molla_solvers::rigid_world::RigidWorld;
use molla_solvers::{FeatherstoneSolver, Solver};

#[cfg(feature = "gpu")]
#[path = "common/gpu_readback.rs"]
mod gpu_readback;

const DT: f64 = 0.00001;
const STEPS: usize = 8;

struct MobileCase {
    kind: JointType,
    body: usize,
    coordinate: usize,
    dof: usize,
    angular: usize,
    axis: usize,
    frame: Quat,
    position: Vec3,
    acceleration: Vec3,
    com_acceleration: Vec3,
    translation_acceleration: Vec3,
}

fn mobile_fixture() -> (Model, State, Control, Vec<MobileCase>) {
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), Vec3::ZERO);
    let axes = [Vec3::X, Vec3::Y, Vec3::Z];
    let force = Vec3::new(0.4, -0.3, 0.2);
    let torque = Vec3::new(3., 2., -1.);
    let mut cases = Vec::new();
    for kind in [JointType::Ball, JointType::D6, JointType::Distance] {
        for frame in [
            Quat::IDENTITY,
            Quat::from_euler(glam::EulerRot::XYZ, 0.2, -0.3, 0.4),
        ] {
            for (axis, local) in axes.into_iter().enumerate() {
                let mass = [1., 17., 83.][axis];
                let inertia = SpatialInertia {
                    mass,
                    com: Vec3::new(0.4, -0.3, 0.2),
                    inertia: Mat3::from_cols(
                        Vec3::new(3., 0.2, 0.3),
                        Vec3::new(0.2, 4., 0.4),
                        Vec3::new(0.3, 0.4, 5.),
                    ) * mass,
                };
                let parent = Transform::new(Vec3::new(1.2, -0.4, 0.8), frame);
                let child = Transform::new(
                    Vec3::new(-0.3, 0.5, -0.9),
                    Quat::from_euler(glam::EulerRot::XYZ, -0.4, 0.3, -0.2),
                );
                let angular = if kind == JointType::Ball { 0 } else { 3 };
                let position = if kind == JointType::Distance {
                    local
                } else {
                    Vec3::ZERO
                };
                let pose = parent
                    .compose(&Transform::new(position, Quat::IDENTITY))
                    .compose(&child.inverse());
                let arm = pose.rotation * (inertia.com - child.position);
                let rotation = Mat3::from_quat(pose.rotation);
                let world_inertia = rotation * inertia.inertia * rotation.transpose();
                let normal = frame * local;
                let (acceleration, com_acceleration) = if kind == JointType::Distance {
                    let inverse = world_inertia.inverse();
                    let free_angular = inverse * torque;
                    let free_com = force / mass;
                    let angular_direction = -arm.cross(normal);
                    let response = 1. / mass + angular_direction.dot(inverse * angular_direction);
                    let reaction = -normal.dot(free_com - free_angular.cross(arm)) / response;
                    (
                        free_angular + inverse * angular_direction * reaction,
                        free_com + normal * (reaction / mass),
                    )
                } else {
                    let anchored = world_inertia
                        + mass
                            * (Mat3::IDENTITY * arm.length_squared()
                                - Mat3::from_cols(arm * arm.x, arm * arm.y, arm * arm.z));
                    let inverse = anchored.inverse();
                    let free = inverse * (torque + arm.cross(force));
                    let acceleration =
                        free - inverse * normal * (normal.dot(free) / normal.dot(inverse * normal));
                    (acceleration, acceleration.cross(arm))
                };
                assert!(acceleration.length() + com_acceleration.length() > 0.01);
                builder.begin_articulation();
                let body = builder.add_body(BodyParams::new().with_spatial_inertia(inertia));
                let joint = match kind {
                    JointType::Ball => builder.add_joint_ball(BodyId::NONE, body, parent, child),
                    JointType::D6 => builder.add_joint_d6(BodyId::NONE, body, parent, child),
                    _ => builder.add_joint_distance(BodyId::NONE, body, parent, child),
                }
                .unwrap();
                let mut lower = vec![f64::NEG_INFINITY; angular + 3];
                let mut upper = vec![f64::INFINITY; angular + 3];
                let mut caps = vec![f64::INFINITY; angular + 3];
                if kind == JointType::Distance {
                    lower[0] = 1.;
                    upper[0] = 1.;
                    caps[axis] = 0.;
                } else {
                    if kind == JointType::D6 {
                        lower[..3].fill(0.);
                        upper[..3].fill(0.);
                    }
                    lower[angular + axis] = 0.;
                    upper[angular + axis] = 0.;
                    caps[angular + axis] = 0.;
                }
                builder.set_joint_limits(joint, &lower, &upper).unwrap();
                builder.set_joint_velocity_limit(joint, &caps).unwrap();
                cases.push(MobileCase {
                    kind,
                    body: body.index(),
                    coordinate: 0,
                    dof: 0,
                    angular,
                    axis,
                    frame,
                    position,
                    acceleration,
                    com_acceleration,
                    translation_acceleration: frame.conjugate()
                        * (com_acceleration - acceleration.cross(arm)),
                });
            }
        }
    }
    let model = builder.build().unwrap();
    assert_eq!(cases.len(), 18);
    let mut initial = model.state().unwrap();
    for (j, case) in cases.iter_mut().enumerate() {
        case.coordinate = model.joint_coord_offset.host().unwrap()[j] as usize;
        case.dof = model.joint_dof_offset.host().unwrap()[j] as usize;
        if case.angular == 3 {
            initial.joint_q.host_mut().unwrap()[case.coordinate..case.coordinate + 3]
                .copy_from_slice(&case.position.to_array());
        }
    }
    molla_sim::eval_fk(&model, &mut initial).unwrap();
    let mut control = model.control();
    control
        .body_external_f
        .host_mut()
        .unwrap()
        .fill(SpatialVector::new(force, torque));
    (model, initial, control, cases)
}

fn verify_mobile(state: &State, residues: &[f32], cases: &[MobileCase], precise: bool) {
    for (j, case) in cases.iter().enumerate() {
        let velocity = state.body_qd.host().unwrap()[case.body];
        let angular_error = (velocity.angular / DT - case.acceleration).length();
        let linear_error = (velocity.linear / DT - case.com_acceleration).length();
        let retained = |i: usize| {
            state.joint_q.host().unwrap()[i] + residues.get(i).copied().unwrap_or(0.) as f64
        };
        let actual = Quat::from_array(std::array::from_fn(|a| {
            retained(case.coordinate + case.angular + a)
        }))
        .normalize();
        let expected =
            Quat::from_scaled_axis(case.frame.conjugate() * case.acceleration * (DT * DT));
        let rotation_error = (actual * expected.conjugate()).xyz().length() * 2.;
        let translation_error = if case.angular == 3 {
            let actual = Vec3::from_array(std::array::from_fn(|a| retained(case.coordinate + a)));
            (actual - (case.position + case.translation_acceleration * (DT * DT))).length()
        } else {
            0.
        };
        let cap_axis = if case.kind == JointType::Distance {
            case.axis
        } else {
            case.angular + case.axis
        };
        let cap_error = state.joint_qd.host().unwrap()[case.dof + cap_axis].abs();
        println!(
            "redundant_mobile joint={j} angular_error={angular_error:.9e} linear_error={linear_error:.9e} rotation_error={rotation_error:.9e} translation_error={translation_error:.9e} cap_error={cap_error:.9e}"
        );
        assert!(
            angular_error.is_finite() && angular_error <= 0.01 * case.acceleration.length().max(1.)
        );
        assert!(
            linear_error.is_finite()
                && linear_error <= 0.01 * case.com_acceleration.length().max(1.)
        );
        let coordinate_tolerance = if precise { 1e-12 } else { 1e-6 };
        assert!(rotation_error.is_finite() && rotation_error <= coordinate_tolerance);
        assert!(translation_error.is_finite() && translation_error <= coordinate_tolerance);
        assert!(cap_error.is_finite() && cap_error <= 1e-9);
    }
}

#[test]
fn cpu_redundant_points_and_caps_preserve_permitted_physical_motion() {
    let (model, initial, control, cases) = mobile_fixture();
    let mut next = model.state().unwrap();
    FeatherstoneSolver::new_cpu()
        .step(
            &model,
            &initial,
            &control,
            &Contacts::with_capacity(0),
            DT,
            &mut next,
        )
        .unwrap();
    verify_mobile(&next, &[], &cases, true);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_redundant_points_and_caps_preserve_permitted_physical_motion() {
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?} dt={DT} mobile_cases=18 masses=1,17,83");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let (mut model, mut initial, mut control, cases) = mobile_fixture();
    model.upload_to_device(device, queue).unwrap();
    control.upload_to_device(device, queue).unwrap();
    initial.upload_to_device(device, queue).unwrap();
    let mut next = model.state().unwrap();
    next.upload_to_device(device, queue).unwrap();
    let mut solver = FeatherstoneSolver::new_gpu(&engine, &model).unwrap();
    solver
        .prepare_gpu_coordinate_compensation(&[&initial, &next])
        .unwrap();
    solver.seed_gpu_coordinate_compensation(&initial).unwrap();
    solver
        .step(
            &model,
            &initial,
            &control,
            &Contacts::with_capacity(0),
            DT,
            &mut next,
        )
        .unwrap();
    assert_eq!(
        gpu_readback::read::<u32>(&engine, solver.gpu_status_buffer().unwrap(), 1)[0],
        0
    );
    next.download_from_device(device, queue).unwrap();
    next.joint_q.download(device, queue).unwrap();
    next.joint_qd.download(device, queue).unwrap();
    let residues = gpu_readback::read::<f32>(
        &engine,
        solver.gpu_coordinate_compensation_buffer(&next).unwrap(),
        model.joint_coord_total,
    );
    verify_mobile(&next, &residues, &cases, true);
}

fn fixture() -> (Model, State, Control) {
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), Vec3::ZERO);
    let mut coordinates = Vec::new();
    for kind in [JointType::Ball, JointType::D6, JointType::Distance] {
        for frame in [
            Quat::IDENTITY,
            Quat::from_euler(glam::EulerRot::XYZ, 0.2, -0.3, 0.4),
        ] {
            for angles in [Vec3::ZERO, Vec3::new(0.3 + 1e-9, -0.6 + 2e-9, 0.2 - 3e-10)] {
                for axis in 0..3 {
                    for pair in [false, true] {
                        for negative in [false, true] {
                            let mass = [1., 17., 83.][axis];
                            let inertia = SpatialInertia {
                                mass,
                                com: Vec3::new(0.4, -0.3, 0.2),
                                inertia: Mat3::from_cols(
                                    Vec3::new(3., 0.2, 0.3),
                                    Vec3::new(0.2, 4., 0.4),
                                    Vec3::new(0.3, 0.4, 5.),
                                ) * mass,
                            };
                            let parent = Transform::new(Vec3::new(1.2, -0.4, 0.8), frame);
                            let child = Transform::new(
                                Vec3::new(-0.3, 0.5, -0.9),
                                Quat::from_euler(glam::EulerRot::XYZ, -0.4, 0.3, -0.2),
                            );
                            builder.begin_articulation();
                            let body =
                                builder.add_body(BodyParams::new().with_spatial_inertia(inertia));
                            let joint = match kind {
                                JointType::Ball => {
                                    builder.add_joint_ball(BodyId::NONE, body, parent, child)
                                }
                                JointType::D6 => {
                                    builder.add_joint_d6(BodyId::NONE, body, parent, child)
                                }
                                _ => builder.add_joint_distance(BodyId::NONE, body, parent, child),
                            }
                            .unwrap();
                            let angular = if kind == JointType::Ball { 0 } else { 3 };
                            let count = angular + 3;
                            let mut lower = vec![f64::NEG_INFINITY; count];
                            let mut upper = vec![f64::INFINITY; count];
                            let mut position = Vec3::ZERO;
                            if kind == JointType::Distance {
                                let radius = if pair { 0. } else { 1. };
                                position[axis] = radius;
                                lower[0] = radius;
                                upper[0] = radius;
                            } else {
                                if kind == JointType::D6 {
                                    lower[..3].fill(0.);
                                    upper[..3].fill(0.);
                                }
                                for a in 0..3 {
                                    if (pair && a != axis) || (!pair && a == axis) {
                                        lower[angular + a] = angles[a];
                                        upper[angular + a] = angles[a];
                                    }
                                }
                            }
                            builder.set_joint_limits(joint, &lower, &upper).unwrap();
                            builder
                                .set_joint_velocity_limit(joint, &vec![0.; count])
                                .unwrap();
                            let rotation =
                                Quat::from_euler(glam::EulerRot::XYZ, angles.x, angles.y, angles.z)
                                    * if negative { -1. } else { 1. };
                            coordinates.push((angular, position, rotation));
                        }
                    }
                }
            }
        }
    }
    let model = builder.build().unwrap();
    assert_eq!(model.joint_count, 144);
    let mut initial = model.state().unwrap();
    for (j, (angular, position, rotation)) in coordinates.into_iter().enumerate() {
        let c = model.joint_coord_offset.host().unwrap()[j] as usize;
        let q = initial.joint_q.host_mut().unwrap();
        if angular == 3 {
            q[c..c + 3].copy_from_slice(&position.to_array());
        }
        q[c + angular..c + angular + 4].copy_from_slice(&rotation.to_array());
    }
    molla_sim::eval_fk(&model, &mut initial).unwrap();
    let mut control = model.control();
    control
        .body_external_f
        .host_mut()
        .unwrap()
        .fill(SpatialVector::new(
            Vec3::new(0.4, -0.3, 0.2),
            Vec3::new(3., 2., -1.),
        ));
    (model, initial, control)
}

fn verify(model: &Model, initial: &State, state: &State, residues: &[f32], step: usize, gpu: bool) {
    for j in 0..model.joint_count {
        let c = model.joint_coord_offset.host().unwrap()[j] as usize;
        let end = model.joint_coord_offset.host().unwrap()[j + 1] as usize;
        let d = model.joint_dof_offset.host().unwrap()[j] as usize;
        let end_d = model.joint_dof_offset.host().unwrap()[j + 1] as usize;
        let coordinate_error = (c..end)
            .map(|i| {
                (state.joint_q.host().unwrap()[i] + residues.get(i).copied().unwrap_or(0.) as f64
                    - initial.joint_q.host().unwrap()[i])
                    .abs()
            })
            .fold(0., f64::max);
        let rate = state.joint_qd.host().unwrap()[d..end_d]
            .iter()
            .map(|v| v.abs())
            .fold(0., f64::max);
        let body = model.joint_child.host().unwrap()[j] as usize;
        let velocity = state.body_qd.host().unwrap()[body];
        let body_rate = velocity.linear.length().max(velocity.angular.length());
        let pose = state.body_q.host().unwrap()[body];
        let reference = initial.body_q.host().unwrap()[body];
        let position_error = (pose.position - reference.position).length();
        let rotation_error = (pose.rotation * reference.rotation.conjugate())
            .xyz()
            .length()
            * 2.;
        println!(
            "redundant_hold step={step} joint={j} kind={} coordinate_error={coordinate_error:.9e} rate={rate:.9e} body_rate={body_rate:.9e} position_error={position_error:.9e} rotation_error={rotation_error:.9e}",
            model.joint_type.host().unwrap()[j]
        );
        assert!(coordinate_error.is_finite() && coordinate_error <= 1e-12);
        assert!(rate.is_finite() && rate <= 1e-9);
        assert!(body_rate.is_finite() && body_rate <= 1e-9);
        let pose_tolerance = if gpu { 1e-3 } else { 1e-8 };
        assert!(position_error.is_finite() && position_error <= pose_tolerance);
        assert!(rotation_error.is_finite() && rotation_error <= pose_tolerance);
    }
}

#[test]
fn cpu_redundant_ball_d6_distance_points_and_zero_caps_hold_under_load() {
    let (model, initial, control) = fixture();
    let mut current = initial.clone_host().unwrap();
    let mut next = model.state().unwrap();
    let mut solver = FeatherstoneSolver::new_cpu();
    for step in 0..STEPS {
        solver
            .step(
                &model,
                &current,
                &control,
                &Contacts::with_capacity(0),
                DT,
                &mut next,
            )
            .unwrap();
        verify(&model, &initial, &next, &[], step, false);
        std::mem::swap(&mut current, &mut next);
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_redundant_ball_d6_distance_points_and_zero_caps_hold_under_load() {
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?} dt={DT} cases=144 steps={STEPS} masses=1,17,83");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let (mut model, initial, mut control) = fixture();
    model.upload_to_device(device, queue).unwrap();
    control.upload_to_device(device, queue).unwrap();
    let mut current = initial.clone_host().unwrap();
    let mut next = model.state().unwrap();
    current.upload_to_device(device, queue).unwrap();
    next.upload_to_device(device, queue).unwrap();
    let mut solver = FeatherstoneSolver::new_gpu(&engine, &model).unwrap();
    solver
        .prepare_gpu_coordinate_compensation(&[&current, &next])
        .unwrap();
    solver.seed_gpu_coordinate_compensation(&current).unwrap();
    for step in 0..STEPS {
        solver
            .step(
                &model,
                &current,
                &control,
                &Contacts::with_capacity(0),
                DT,
                &mut next,
            )
            .unwrap();
        assert_eq!(
            gpu_readback::read::<u32>(&engine, solver.gpu_status_buffer().unwrap(), 1)[0],
            0
        );
        next.download_from_device(device, queue).unwrap();
        next.joint_q.download(device, queue).unwrap();
        next.joint_qd.download(device, queue).unwrap();
        let residues = gpu_readback::read::<f32>(
            &engine,
            solver.gpu_coordinate_compensation_buffer(&next).unwrap(),
            model.joint_coord_total,
        );
        verify(&model, &initial, &next, &residues, step, true);
        std::mem::swap(&mut current, &mut next);
    }
}

struct RuntimeFixture {
    scene: RigidScene,
    bodies: Vec<BodyHandle>,
    joints: Vec<JointHandle>,
    radial: Vec<Option<JointHandle>>,
}

fn runtime_fixture(model: &Model, initial: &State) -> RuntimeFixture {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let mut bodies = Vec::new();
    let mut joints = Vec::new();
    let mut radial = Vec::new();
    let axes = [
        JointAxis::LinX,
        JointAxis::LinY,
        JointAxis::LinZ,
        JointAxis::AngX,
        JointAxis::AngY,
        JointAxis::AngZ,
    ];
    for j in 0..model.joint_count {
        let b = model.joint_child.host().unwrap()[j] as usize;
        let body = scene
            .insert_body(BodyDesc {
                params: BodyParams::new().with_pose(initial.body_q.host().unwrap()[b]),
                additional_mass: Some(MassProperties {
                    mass: model.body_mass.host().unwrap()[b],
                    local_com: model.body_com.host().unwrap()[b],
                    inertia: model.body_inertia.host().unwrap()[b],
                }),
                ..Default::default()
            })
            .unwrap();
        let kind = [JointType::Ball, JointType::D6, JointType::Distance]
            .into_iter()
            .find(|&kind| kind as u32 == model.joint_type.host().unwrap()[j])
            .unwrap();
        let joint = scene
            .insert_joint(JointDesc {
                parent: None,
                child: body,
                kind: if kind == JointType::Distance {
                    JointType::D6
                } else {
                    kind
                },
                axis: Vec3::X,
                frame_parent: model.joint_x_p.host().unwrap()[j],
                frame_child: model.joint_x_c.host().unwrap()[j],
            })
            .unwrap();
        let dof = model.joint_dof_offset.host().unwrap()[j] as usize;
        let lower = model.joint_limit_lower.host().unwrap();
        let upper = model.joint_limit_upper.host().unwrap();
        let caps = model.joint_velocity_limit.host().unwrap();
        if kind == JointType::Distance {
            let distance = scene
                .insert_joint(JointDesc {
                    parent: None,
                    child: body,
                    kind,
                    axis: Vec3::X,
                    frame_parent: model.joint_x_p.host().unwrap()[j],
                    frame_child: model.joint_x_c.host().unwrap()[j],
                })
                .unwrap();
            scene
                .set_distance_limits(
                    distance,
                    DistanceLimits {
                        min: lower[dof].is_finite().then_some(lower[dof]),
                        max: upper[dof].is_finite().then_some(upper[dof]),
                    },
                )
                .unwrap();
            radial.push(Some(distance));
        } else {
            radial.push(None);
        }
        let angular = if kind == JointType::Ball { 3 } else { 0 };
        for (d, &axis) in axes[angular..].iter().enumerate() {
            if kind != JointType::Distance && lower[dof + d].is_finite() {
                scene
                    .set_joint_limits(joint, axis, Some([lower[dof + d], upper[dof + d]]))
                    .unwrap();
            }
            if caps[dof + d].is_finite() {
                scene
                    .set_joint_velocity_limit(joint, axis, Some(caps[dof + d]))
                    .unwrap();
            }
        }
        bodies.push(body);
        joints.push(joint);
    }
    RuntimeFixture {
        scene,
        bodies,
        joints,
        radial,
    }
}

fn load_runtime(world: &mut RigidWorld, bodies: &[BodyHandle], control: &Control) {
    for (&body, &wrench) in bodies.iter().zip(control.body_external_f.host().unwrap()) {
        world.scene.reset_body_forces(body).unwrap();
        world.scene.add_body_wrench(body, wrench).unwrap();
    }
}

#[allow(clippy::too_many_arguments)]
fn verify_runtime_hold(
    mut world: RigidWorld,
    bodies: &[BodyHandle],
    joints: &[JointHandle],
    radial: &[Option<JointHandle>],
    initial: &State,
    control: &Control,
    gpu: bool,
) {
    world.solver.substeps = 1;
    world.set_reaction_readout(true).unwrap();
    for step in 0..STEPS {
        load_runtime(&mut world, bodies, control);
        world.step(DT).unwrap();
        assert!(world.quarantined_bodies().is_empty());
        assert!(world.joint_break_events().is_empty());
        for (j, (&body, &joint)) in bodies.iter().zip(joints).enumerate() {
            let pose = world.scene.body_pose(body).unwrap();
            let velocity = world.scene.body_velocity(body).unwrap();
            let expected = initial.body_q.host().unwrap()[j];
            let position_error = (pose.position - expected.position).length();
            let rotation_error = (pose.rotation * expected.rotation.conjugate())
                .xyz()
                .length()
                * 2.;
            let body_rate = velocity.linear.length().max(velocity.angular.length());
            let b = world.scene.body_index(body).unwrap().index();
            let arm = pose.rotation
                * (world.scene.model().body_com.host().unwrap()[b]
                    - world.scene.joint(joint).unwrap().frame_child.position);
            let load = control.body_external_f.host().unwrap()[j];
            let mut reaction = *world.joint_reaction(joint).unwrap();
            if let Some(distance) = radial[j] {
                let external = world.joint_reaction(distance).unwrap();
                reaction.child = reaction.child + external.child;
                reaction.parent = reaction.parent + external.parent;
            }
            let force_error = (reaction.child.linear + load.linear).length();
            let torque_error =
                (reaction.child.angular + load.angular + arm.cross(load.linear)).length();
            println!(
                "redundant_runtime_hold step={step} joint={j} position_error={position_error:.9e} rotation_error={rotation_error:.9e} body_rate={body_rate:.9e} force_error={force_error:.9e} torque_error={torque_error:.9e}"
            );
            let tolerance = if gpu { 1e-3 } else { 1e-8 };
            assert!(position_error.is_finite() && position_error <= tolerance);
            assert!(rotation_error.is_finite() && rotation_error <= tolerance);
            assert!(body_rate.is_finite() && body_rate <= 1e-9);
            assert!(force_error.is_finite() && force_error <= 0.01);
            assert!(
                torque_error.is_finite()
                    && torque_error
                        <= 0.01 * (load.angular + arm.cross(load.linear)).length().max(1.)
            );
            assert!((reaction.child.linear + reaction.parent.linear).length() <= tolerance);
        }
    }
}

#[test]
fn cpu_runtime_redundant_points_and_zero_caps_hold_under_load() {
    let (model, initial, control) = fixture();
    let RuntimeFixture {
        scene,
        bodies,
        joints,
        radial,
    } = runtime_fixture(&model, &initial);
    verify_runtime_hold(
        RigidWorld::from_scene(scene).unwrap(),
        &bodies,
        &joints,
        &radial,
        &initial,
        &control,
        false,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_runtime_redundant_points_and_zero_caps_hold_under_load() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?} dt={DT} runtime_cases=144 steps={STEPS}");
    let (model, initial, control) = fixture();
    let RuntimeFixture {
        scene,
        bodies,
        joints,
        radial,
    } = runtime_fixture(&model, &initial);
    verify_runtime_hold(
        RigidWorld::from_scene_gpu(scene, engine).unwrap(),
        &bodies,
        &joints,
        &radial,
        &initial,
        &control,
        true,
    );
}

fn verify_runtime_mobile(
    mut world: RigidWorld,
    bodies: &[BodyHandle],
    joints: &[JointHandle],
    control: &Control,
    cases: &mut [MobileCase],
    gpu: bool,
) {
    world.solver.substeps = 1;
    for ((case, &body), &joint) in cases.iter_mut().zip(bodies).zip(joints) {
        case.body = world.scene.body_index(body).unwrap().index();
        let j = world.scene.joint_index(joint).unwrap().index();
        case.coordinate = world.scene.model().joint_coord_offset.host().unwrap()[j] as usize;
        case.dof = world.scene.model().joint_dof_offset.host().unwrap()[j] as usize;
    }
    load_runtime(&mut world, bodies, control);
    world.step(DT).unwrap();
    assert!(world.quarantined_bodies().is_empty());
    verify_mobile(world.scene.state(), &[], cases, !gpu);
}

#[test]
fn cpu_runtime_redundant_points_and_caps_preserve_permitted_physical_motion() {
    let (model, initial, control, mut cases) = mobile_fixture();
    let RuntimeFixture {
        scene,
        bodies,
        joints,
        ..
    } = runtime_fixture(&model, &initial);
    verify_runtime_mobile(
        RigidWorld::from_scene(scene).unwrap(),
        &bodies,
        &joints,
        &control,
        &mut cases,
        false,
    );
}

fn incompatible_radial_point_scene() -> (RigidScene, BodyHandle) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let body = scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_pose(Transform::from_translation(Vec3::X * (1. + 1e-7))),
            additional_mass: Some(MassProperties {
                mass: 1.,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap();
    let tree = scene
        .insert_joint(JointDesc {
            parent: None,
            child: body,
            kind: JointType::D6,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    for axis in JointAxis::ALL {
        scene
            .set_joint_velocity_limit(tree, axis, Some(0.))
            .unwrap();
    }
    let distance = scene
        .insert_joint(JointDesc {
            parent: None,
            child: body,
            kind: JointType::Distance,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    scene
        .set_distance_limits(
            distance,
            DistanceLimits {
                min: Some(1.),
                max: Some(1.),
            },
        )
        .unwrap();
    (scene, body)
}

fn verify_incompatible_radial_point(mut world: RigidWorld, body: BodyHandle) {
    let pose = world.scene.body_pose(body).unwrap();
    let velocity = world.scene.body_velocity(body).unwrap();
    world.solver.substeps = 1;
    let error = world.step(DT).unwrap_err();
    println!("incompatible_runtime error={error:?}");
    assert!(matches!(error, molla_core::Error::Solver(_)));
    assert_eq!(world.completed_steps(), 0);
    assert_eq!(world.scene.body_pose(body).unwrap(), pose);
    assert_eq!(world.scene.body_velocity(body).unwrap(), velocity);
    assert!(world.joint_break_events().is_empty());
}

#[test]
fn cpu_runtime_incompatible_radial_point_and_zero_caps_fail_without_advancing() {
    let (scene, body) = incompatible_radial_point_scene();
    verify_incompatible_radial_point(RigidWorld::from_scene(scene).unwrap(), body);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_runtime_incompatible_radial_point_and_zero_caps_fail_without_advancing() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?} incompatible_radial_caps=true");
    let (scene, body) = incompatible_radial_point_scene();
    assert!(!scene.model().requires_angular_lock_rows().unwrap());
    verify_incompatible_radial_point(RigidWorld::from_scene_gpu(scene, engine).unwrap(), body);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_runtime_redundant_points_and_caps_preserve_permitted_physical_motion() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?} dt={DT} runtime_mobile_cases=18");
    let (model, initial, control, mut cases) = mobile_fixture();
    let RuntimeFixture {
        scene,
        bodies,
        joints,
        ..
    } = runtime_fixture(&model, &initial);
    verify_runtime_mobile(
        RigidWorld::from_scene_gpu(scene, engine).unwrap(),
        &bodies,
        &joints,
        &control,
        &mut cases,
        true,
    );
}
