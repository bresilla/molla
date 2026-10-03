use molla_core::{BodyId, WorldId};
use molla_math::{Mat3, Quat, SpatialInertia, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::JointAxis;
use molla_sim::runtime::{BodyDesc, JointDesc, JointHandle, RigidScene};
use molla_sim::{BodyParams, Contacts, Control, Model, ModelBuilder, State};
use molla_sim::{JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;
use molla_solvers::{FeatherstoneSolver, Solver};

#[cfg(feature = "gpu")]
#[path = "common/gpu_readback.rs"]
mod gpu_readback;

const DT: f64 = 0.00001;

struct Case {
    body: usize,
    coordinate: usize,
    dof: usize,
    lower: [f64; 3],
    upper: [f64; 3],
    cap: Option<usize>,
    initial: Quat,
    parent: Transform,
    child: Transform,
    acceleration: Vec3,
    com_acceleration: Vec3,
    reaction: SpatialVector,
}

fn fixture() -> (Model, State, Control, Vec<Case>) {
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), Vec3::ZERO);
    let inertia = SpatialInertia {
        mass: 2.,
        com: Vec3::new(0.4, -0.3, 0.2),
        inertia: Mat3::from_cols(
            Vec3::new(3., 0.2, 0.3),
            Vec3::new(0.2, 4., 0.4),
            Vec3::new(0.3, 0.4, 5.),
        ),
    };
    let child = Transform::new(
        Vec3::new(-0.3, 0.5, -0.9),
        Quat::from_euler(glam::EulerRot::XYZ, -0.4, 0.3, -0.2),
    );
    let axes = [Vec3::X, Vec3::Y, Vec3::Z];
    let mut cases = Vec::new();
    let mut loads = Vec::new();
    for frame in [
        Quat::IDENTITY,
        Quat::from_euler(glam::EulerRot::XYZ, 0.2, -0.3, 0.4),
    ] {
        let parent = Transform::new(Vec3::new(1.2, -0.4, 0.8), frame);
        for pitch in [
            -0.6,
            -(std::f64::consts::PI - 0.6),
            0.6,
            std::f64::consts::PI - 0.6,
        ] {
            let angles = Vec3::new(0.3 + 1e-9, pitch + 2e-9, 0.2 - 3e-10);
            let gradients = [
                Vec3::new(
                    1.,
                    angles.x.sin() * angles.y.tan(),
                    -angles.x.cos() * angles.y.tan(),
                ),
                Vec3::new(0., angles.x.cos(), angles.x.sin()),
                Vec3::new(
                    0.,
                    -angles.x.sin() / angles.y.cos(),
                    angles.x.cos() / angles.y.cos(),
                ),
            ];
            let tangents = [
                Vec3::X,
                Quat::from_rotation_x(angles.x) * Vec3::Y,
                Quat::from_rotation_x(angles.x) * Quat::from_rotation_y(angles.y) * Vec3::Z,
            ];
            for mode in 0..4 {
                for axis in 0..3 {
                    for upper in [false, true] {
                        for outward in [false, true] {
                            builder.begin_articulation();
                            let body =
                                builder.add_body(BodyParams::new().with_spatial_inertia(inertia));
                            let joint = builder
                                .add_joint_ball(BodyId::NONE, body, parent, child)
                                .unwrap();
                            let mut lower = [f64::NEG_INFINITY; 3];
                            let mut higher = [f64::INFINITY; 3];
                            if mode == 0 {
                                lower[axis] = angles[axis];
                                higher[axis] = angles[axis];
                            } else if mode == 1 {
                                for k in 0..3 {
                                    if k != axis {
                                        lower[k] = angles[k];
                                        higher[k] = angles[k];
                                    }
                                }
                            } else if mode == 2 {
                                lower[axis] = angles[axis] - if upper { 1e-9 } else { 0. };
                                higher[axis] = angles[axis] + if upper { 0. } else { 1e-9 };
                                assert_eq!(lower[axis] as f32, higher[axis] as f32);
                            } else {
                                let mut caps = [f64::INFINITY; 3];
                                caps[axis] = 0.;
                                builder.set_joint_velocity_limit(joint, &caps).unwrap();
                            }
                            builder.set_joint_limits(joint, &lower, &higher).unwrap();
                            let initial =
                                Quat::from_euler(glam::EulerRot::XYZ, angles.x, angles.y, angles.z)
                                    * if upper { -1. } else { 1. };
                            let pose = parent
                                .compose(&Transform::new(Vec3::ZERO, initial))
                                .compose(&child.inverse());
                            let arm = pose.rotation * (inertia.com - child.position);
                            let rotation = Mat3::from_quat(pose.rotation);
                            let world_inertia = rotation * inertia.inertia * rotation.transpose();
                            let anchored = world_inertia
                                + inertia.mass
                                    * (Mat3::IDENTITY * arm.length_squared()
                                        - Mat3::from_cols(arm * arm.x, arm * arm.y, arm * arm.z));
                            let tangent = frame
                                * if mode == 3 {
                                    axes[axis]
                                } else {
                                    tangents[axis]
                                };
                            let normal = frame
                                * if mode == 3 {
                                    axes[axis]
                                } else {
                                    gradients[axis]
                                };
                            let direction =
                                if upper { 1. } else { -1. } * if outward { 1. } else { -1. };
                            let free = tangent * (2. * direction);
                            let mut torque = anchored * free;
                            let acceleration = if mode == 1 {
                                let transverse =
                                    (frame * axes[(axis + 1) % 3]).cross(tangent).normalize();
                                torque += transverse * 3.;
                                tangent * (tangent.dot(torque) / tangent.dot(anchored * tangent))
                            } else if mode == 0 || mode == 3 || outward {
                                let inverse = anchored.inverse();
                                free - inverse
                                    * normal
                                    * (normal.dot(free) / normal.dot(inverse * normal))
                            } else {
                                free
                            };
                            cases.push(Case {
                                body: body.index(),
                                coordinate: 0,
                                dof: 0,
                                lower,
                                upper: higher,
                                cap: (mode == 3).then_some(axis),
                                initial,
                                parent,
                                child,
                                acceleration,
                                com_acceleration: acceleration.cross(arm),
                                reaction: SpatialVector::new(
                                    inertia.mass * acceleration.cross(arm),
                                    world_inertia * acceleration - torque
                                        + arm.cross(inertia.mass * acceleration.cross(arm)),
                                ),
                            });
                            loads.push(SpatialVector::new(Vec3::ZERO, torque));
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases.len(), 384);
    let model = builder.build().unwrap();
    let mut initial = model.state().unwrap();
    for (j, case) in cases.iter_mut().enumerate() {
        case.coordinate = model.joint_coord_offset.host().unwrap()[j] as usize;
        case.dof = model.joint_dof_offset.host().unwrap()[j] as usize;
        initial.joint_q.host_mut().unwrap()[case.coordinate..case.coordinate + 4]
            .copy_from_slice(&case.initial.to_array());
        initial.body_f.host_mut().unwrap()[case.body] = loads[j];
    }
    molla_sim::eval_fk(&model, &mut initial).unwrap();
    let control = model.control();
    (model, initial, control, cases)
}

fn verify(state: &State, residues: &[f32], cases: &[Case], gpu: bool, retained: bool) {
    for (j, case) in cases.iter().enumerate() {
        let rate = state.body_qd.host().unwrap()[case.body];
        let angular_error = (rate.angular / DT - case.acceleration).length();
        if j == 0 {
            println!(
                "ball_vectors actual={:?} expected={:?}",
                rate.angular / DT,
                case.acceleration
            );
        }
        let linear_error = (rate.linear / DT - case.com_acceleration).length();
        let actual = Quat::from_array(std::array::from_fn(|axis| {
            let c = case.coordinate + axis;
            state.joint_q.host().unwrap()[c] + residues.get(c).copied().unwrap_or(0.) as f64
        }))
        .normalize();
        let expected = Quat::from_scaled_axis(
            case.parent.rotation.conjugate() * case.acceleration * (DT * DT),
        ) * case.initial;
        let rotation_error = (actual * expected.conjugate()).xyz().length() * 2.;
        let pose = state.body_q.host().unwrap()[case.body];
        let anchor_error =
            (pose.transform_point(case.child.position) - case.parent.position).length();
        let (x, y, z) = actual.to_euler(glam::EulerRot::XYZ);
        let violation = [
            [x, y, z],
            [
                x + std::f64::consts::PI,
                std::f64::consts::PI - y,
                z + std::f64::consts::PI,
            ],
        ]
        .into_iter()
        .map(|candidate| {
            (0..3)
                .filter(|&axis| case.lower[axis].is_finite() || case.upper[axis].is_finite())
                .map(|axis| {
                    let reference = if case.lower[axis].is_finite() {
                        case.lower[axis]
                    } else {
                        case.upper[axis]
                    };
                    let angle = candidate[axis]
                        + std::f64::consts::TAU
                            * ((reference - candidate[axis]) / std::f64::consts::TAU).round();
                    (case.lower[axis] - angle)
                        .max(angle - case.upper[axis])
                        .max(0.)
                })
                .fold(0., f64::max)
        })
        .fold(f64::INFINITY, f64::min);
        let cap_error = case.cap.map_or(0., |axis| {
            state.joint_qd.host().unwrap()[case.dof + axis].abs()
        });
        println!(
            "ball_angular_response case={j} angular_error={angular_error:.9e} linear_error={linear_error:.9e} rotation_error={rotation_error:.9e} anchor_error={anchor_error:.9e} violation={violation:.9e} cap_error={cap_error:.9e}"
        );
        assert!(angular_error <= 0.01 * case.acceleration.length().max(1.));
        assert!(linear_error <= 0.01 * case.com_acceleration.length().max(1.));
        if retained {
            assert!(rotation_error.is_finite() && rotation_error <= 1e-12);
        }
        assert!(anchor_error <= if gpu { 1e-3 } else { 1e-8 });
        if retained {
            assert!(violation.is_finite() && violation <= 1e-12);
        }
        assert!(cap_error.is_finite() && cap_error <= 1e-9);
    }
}

#[test]
fn cpu_raw_ball_partial_points_intervals_and_caps_use_physical_angular_rows() {
    let (model, initial, control, cases) = fixture();
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
    verify(&next, &[], &cases, false, true);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_raw_ball_partial_points_intervals_and_caps_use_physical_angular_rows() {
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?} dt={DT} cases=384 width=1e-9");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let (mut model, mut initial, mut control, cases) = fixture();
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
    verify(&next, &residues, &cases, true, true);
}

fn runtime_fixture() -> (RigidScene, Vec<Case>, Vec<JointHandle>) {
    let (model, initial, _, mut cases) = fixture();
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let mut joints = Vec::new();
    let mut bodies = Vec::new();
    for case in &cases {
        let body = scene
            .insert_body(BodyDesc {
                params: BodyParams::new().with_pose(initial.body_q.host().unwrap()[case.body]),
                additional_mass: Some(MassProperties {
                    mass: model.body_mass.host().unwrap()[case.body],
                    local_com: model.body_com.host().unwrap()[case.body],
                    inertia: model.body_inertia.host().unwrap()[case.body],
                }),
                ..Default::default()
            })
            .unwrap();
        let joint = scene
            .insert_joint(JointDesc {
                parent: None,
                child: body,
                kind: JointType::Ball,
                axis: Vec3::X,
                frame_parent: case.parent,
                frame_child: case.child,
            })
            .unwrap();
        for (axis, name) in [JointAxis::AngX, JointAxis::AngY, JointAxis::AngZ]
            .into_iter()
            .enumerate()
        {
            if case.lower[axis].is_finite() && case.upper[axis].is_finite() {
                scene
                    .set_joint_limits(joint, name, Some([case.lower[axis], case.upper[axis]]))
                    .unwrap();
            }
            if case.cap == Some(axis) {
                scene
                    .set_joint_velocity_limit(joint, name, Some(0.))
                    .unwrap();
            }
        }
        scene
            .add_body_wrench(body, initial.body_f.host().unwrap()[case.body])
            .unwrap();
        bodies.push(body);
        joints.push(joint);
    }
    for ((case, body), joint) in cases.iter_mut().zip(bodies).zip(&joints) {
        case.body = scene.body_index(body).unwrap().index();
        let j = scene.joint_index(*joint).unwrap().index();
        case.coordinate = scene.model().joint_coord_offset.host().unwrap()[j] as usize;
        case.dof = scene.model().joint_dof_offset.host().unwrap()[j] as usize;
    }
    (scene, cases, joints)
}

fn verify_runtime(mut world: RigidWorld, cases: &[Case], joints: &[JointHandle], gpu: bool) {
    world.solver.substeps = 1;
    world.set_reaction_readout(true).unwrap();
    world.step(DT).unwrap();
    assert!(world.quarantined_bodies().is_empty());
    assert!(world.joint_break_events().is_empty());
    verify(world.scene.state(), &[], cases, gpu, !gpu);
    for (j, (case, joint)) in cases.iter().zip(joints).enumerate() {
        let actual = world.joint_reaction(*joint).unwrap();
        let force_error = (actual.child.linear - case.reaction.linear).length();
        let torque_error = (actual.child.angular - case.reaction.angular).length();
        println!(
            "ball_runtime_reaction case={j} force_error={force_error:.9e} torque_error={torque_error:.9e}"
        );
        assert!(force_error <= 0.01 * case.reaction.linear.length().max(1.));
        assert!(torque_error <= 0.01 * case.reaction.angular.length().max(1.));
        assert!(
            (actual.child.linear + actual.parent.linear).length() <= if gpu { 1e-3 } else { 1e-8 }
        );
        assert!(
            (actual.child.angular + actual.parent.angular).length()
                <= if gpu { 1e-3 } else { 1e-8 }
        );
    }
}

#[test]
fn cpu_runtime_ball_partial_limits_and_caps_preserve_physical_response_and_reactions() {
    let (scene, cases, joints) = runtime_fixture();
    verify_runtime(
        RigidWorld::from_scene(scene).unwrap(),
        &cases,
        &joints,
        false,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_runtime_ball_partial_limits_and_caps_preserve_physical_response_and_reactions() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?} dt={DT} cases=384 width=1e-9 runtime=true");
    let (scene, cases, joints) = runtime_fixture();
    verify_runtime(
        RigidWorld::from_scene_gpu(scene, engine).unwrap(),
        &cases,
        &joints,
        true,
    );
}
