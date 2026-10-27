use molla_core::{BodyId, WorldId};
use molla_math::{Mat3, Quat, SpatialInertia, Transform, Vec3};
use molla_sim::{BodyParams, Contacts, Control, Model, ModelBuilder, State};
use molla_solvers::{FeatherstoneSolver, Solver};

#[cfg(feature = "gpu")]
#[path = "common/gpu_readback.rs"]
mod gpu_readback;

const DT: f64 = 0.00005;
const WIDTH: f64 = 1e-8;

struct Case {
    coordinate: usize,
    dof: usize,
    lower: f64,
    upper: f64,
    expected_rate: f64,
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
    let parent = Transform::new(
        Vec3::new(1.2, -0.4, 0.8),
        Quat::from_euler(glam::EulerRot::XYZ, 0.2, -0.3, 0.4),
    );
    let child = Transform::new(
        Vec3::new(-0.3, 0.5, -0.9),
        Quat::from_euler(glam::EulerRot::XYZ, -0.4, 0.3, -0.2),
    );
    let axis = Vec3::new(1., 2., -1.).normalize();
    let body_axis = child.rotation * axis;
    let arm = inertia.com - child.position;
    let revolute_mass = body_axis.dot(inertia.inertia * body_axis)
        + inertia.mass * body_axis.cross(arm).length_squared();
    let mut cases = Vec::new();
    let mut initial_coordinates = Vec::new();
    let mut efforts = Vec::new();
    for revolute in [false, true] {
        for point in [false, true] {
            for upper_stop in [false, true] {
                for middle in [false, true] {
                    if point && middle {
                        continue;
                    }
                    for outward in [false, true] {
                        builder.begin_articulation();
                        let body =
                            builder.add_body(BodyParams::new().with_spatial_inertia(inertia));
                        let joint = if revolute {
                            builder.add_joint_revolute(BodyId::NONE, body, parent, child, axis)
                        } else {
                            builder.add_joint_prismatic(BodyId::NONE, body, parent, child, axis)
                        }
                        .unwrap();
                        let endpoint = 1.5 + 1e-9;
                        let (lower, upper) = if point {
                            (endpoint, endpoint)
                        } else if upper_stop {
                            (endpoint - WIDTH, endpoint)
                        } else {
                            (endpoint, endpoint + WIDTH)
                        };
                        assert_eq!(lower as f32, upper as f32);
                        builder.set_joint_limits(joint, &[lower], &[upper]).unwrap();
                        let direction =
                            if upper_stop { 1. } else { -1. } * if outward { 1. } else { -1. };
                        let rate = if point || (!middle && outward) {
                            0.
                        } else {
                            direction * DT
                        };
                        let initial = if middle {
                            0.5 * lower + 0.5 * upper
                        } else if upper_stop {
                            upper
                        } else {
                            lower
                        };
                        cases.push(Case {
                            coordinate: cases.len(),
                            dof: cases.len(),
                            lower,
                            upper,
                            expected_rate: rate,
                        });
                        initial_coordinates.push(initial);
                        efforts.push(
                            direction
                                * if revolute {
                                    revolute_mass
                                } else {
                                    inertia.mass
                                },
                        );
                    }
                }
            }
        }
    }
    let model = builder.build().unwrap();
    let mut initial = model.state().unwrap();
    initial
        .joint_q
        .host_mut()
        .unwrap()
        .copy_from_slice(&initial_coordinates);
    let mut control = model.control();
    control
        .joint_force
        .host_mut()
        .unwrap()
        .copy_from_slice(&efforts);
    assert_eq!(cases.len(), 24);
    (model, initial, control, cases)
}

fn verify(next: &State, residues: &[f32], cases: &[Case]) {
    for (index, case) in cases.iter().enumerate() {
        let coordinate = next.joint_q.host().unwrap()[case.coordinate]
            + residues.get(case.coordinate).copied().unwrap_or(0.) as f64;
        let rate = next.joint_qd.host().unwrap()[case.dof];
        let acceleration_error = (rate - case.expected_rate).abs() / DT;
        let violation = (case.lower - coordinate)
            .max(coordinate - case.upper)
            .max(0.);
        println!(
            "scalar_precision case={index} acceleration_error={acceleration_error:.9e} violation={violation:.9e}"
        );
        assert!(acceleration_error.is_finite() && acceleration_error <= 0.01);
        assert!(violation.is_finite() && violation <= 1e-12);
    }
}

#[test]
fn cpu_authored_scalar_intervals_preserve_motion_and_stop_response() {
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
    verify(&next, &[], &cases);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_authored_scalar_intervals_preserve_motion_and_stop_response() {
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?} dt={DT} width={WIDTH}");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let (mut model, mut initial, mut control, cases) = fixture();
    model.upload_to_device(device, queue).unwrap();
    initial.upload_to_device(device, queue).unwrap();
    control.upload_to_device(device, queue).unwrap();
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
    verify(&next, &residues, &cases);
}

fn point_fixture(ball: bool) -> (Model, State, Control) {
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), Vec3::ZERO);
    let mut coordinates = Vec::new();
    let mut rates = Vec::new();
    for frame in [
        Quat::IDENTITY,
        Quat::from_euler(glam::EulerRot::XYZ, 0.2, -0.3, 0.4),
    ] {
        for pitch in [
            -0.6,
            0.6,
            -3. * std::f64::consts::FRAC_PI_2,
            3. * std::f64::consts::FRAC_PI_2,
        ] {
            for negative in [false, true] {
                for rate in [-3., 0., 3.] {
                    for perturbed in [false, true] {
                        builder.begin_articulation();
                        let body = builder.add_body(BodyParams::new().with_spatial_inertia(
                            SpatialInertia {
                                mass: 2.,
                                com: Vec3::new(0.4, -0.3, 0.2),
                                inertia: Mat3::from_cols(
                                    Vec3::new(3., 0.2, 0.3),
                                    Vec3::new(0.2, 4., 0.4),
                                    Vec3::new(0.3, 0.4, 5.),
                                ),
                            },
                        ));
                        let parent = Transform::new(Vec3::new(1.2, -0.4, 0.8), frame);
                        let child = Transform::new(
                            Vec3::new(-0.3, 0.5, -0.9),
                            Quat::from_euler(glam::EulerRot::XYZ, -0.4, 0.3, -0.2),
                        );
                        let joint = if ball {
                            builder.add_joint_ball(BodyId::NONE, body, parent, child)
                        } else {
                            builder.add_joint_d6(BodyId::NONE, body, parent, child)
                        }
                        .unwrap();
                        let target = [1.5 + 1e-9, -2.5 - 2e-9, 0.125 + 3e-10, 0.3, pitch, 0.2];
                        let bounds = if ball { &target[3..] } else { &target[..] };
                        builder.set_joint_limits(joint, bounds, bounds).unwrap();
                        let translation = Vec3::from_slice(&target[..3])
                            + if perturbed {
                                Vec3::new(0.2, -0.3, 0.4)
                            } else {
                                Vec3::ZERO
                            };
                        let rotation =
                            Quat::from_euler(glam::EulerRot::XYZ, target[3], target[4], target[5]);
                        let rotation = if perturbed {
                            Quat::from_rotation_x(0.1) * rotation
                        } else {
                            rotation
                        } * if negative { -1. } else { 1. };
                        if !ball {
                            coordinates.extend(translation.to_array());
                        }
                        coordinates.extend(rotation.to_array());
                        if !ball {
                            rates.extend([rate, -rate, rate]);
                        }
                        rates.extend([-rate, rate, -rate]);
                    }
                }
            }
        }
    }
    let model = builder.build().unwrap();
    assert_eq!(model.joint_count, 96);
    let mut initial = model.state().unwrap();
    initial
        .joint_q
        .host_mut()
        .unwrap()
        .copy_from_slice(&coordinates);
    initial.joint_qd.host_mut().unwrap().copy_from_slice(&rates);
    let mut control = model.control();
    control.joint_force.host_mut().unwrap().fill(8.);
    (model, initial, control)
}

fn edit_point_targets(model: &mut Model) {
    let mut targets = model.joint_limit_lower.host().unwrap().to_vec();
    for (index, value) in targets.iter_mut().enumerate() {
        *value += (index % 6 + 1) as f64 * 1e-9;
    }
    model
        .joint_limit_lower
        .host_mut()
        .unwrap()
        .copy_from_slice(&targets);
    model
        .joint_limit_upper
        .host_mut()
        .unwrap()
        .copy_from_slice(&targets);
}

fn verify_points(model: &Model, state: &State, residues: &[f32], gpu: bool) {
    for joint in 0..model.joint_count {
        let c = model.joint_coord_offset.host().unwrap()[joint] as usize;
        let d = model.joint_dof_offset.host().unwrap()[joint] as usize;
        let ball = model.joint_type.host().unwrap()[joint] == molla_sim::JointType::Ball as u32;
        let offset = if ball { 0 } else { 3 };
        let count = if ball { 3 } else { 6 };
        let target = &model.joint_limit_lower.host().unwrap()[d..d + count];
        let coordinates = state.joint_q.host().unwrap();
        let retained: [f64; 4] = std::array::from_fn(|axis| {
            let index = c + offset + axis;
            coordinates[index] + residues.get(index).copied().unwrap_or(0.) as f64
        });
        let translation = if ball {
            Vec3::ZERO
        } else {
            Vec3::from_slice(&target[..3])
        };
        let translation_error = if ball {
            0.
        } else {
            let actual = Vec3::from_array(std::array::from_fn(|axis| {
                coordinates[c + axis] + residues.get(c + axis).copied().unwrap_or(0.) as f64
            }));
            (actual - translation).length()
        };
        let expected_rotation = Quat::from_euler(
            glam::EulerRot::XYZ,
            target[offset],
            target[offset + 1],
            target[offset + 2],
        )
        .normalize();
        let actual_rotation = Quat::from_array(retained).normalize();
        let rotation_error = (actual_rotation * expected_rotation.conjugate())
            .xyz()
            .length()
            * 2.;
        println!(
            "quaternion_point_publication ball={ball} joint={joint} translation_error={translation_error:.9e} rotation_error={rotation_error:.9e}"
        );
        assert!(translation_error.is_finite() && translation_error <= 1e-12);
        assert!(rotation_error.is_finite() && rotation_error <= 1e-12);
        assert!(
            state.joint_qd.host().unwrap()[d..d + count]
                .iter()
                .all(|&value| value == 0.)
        );
        let local = Transform::new(translation, expected_rotation);
        let expected_pose = model.joint_x_p.host().unwrap()[joint]
            .compose(&local)
            .compose(&model.joint_x_c.host().unwrap()[joint].inverse());
        let body = model.joint_child.host().unwrap()[joint] as usize;
        let pose = state.body_q.host().unwrap()[body];
        let tolerance = if gpu { 1e-3 } else { 1e-12 };
        assert!((pose.position - expected_pose.position).length() <= tolerance);
        assert!(
            (pose.rotation * expected_pose.rotation.conjugate())
                .xyz()
                .length()
                * 2.
                <= tolerance
        );
        assert!(state.body_qd.host().unwrap()[body].linear.length() <= 1e-9);
        assert!(state.body_qd.host().unwrap()[body].angular.length() <= 1e-9);
    }
}

#[test]
fn cpu_d6_points_publish_authored_state_after_bound_edits() {
    point_publication_cpu(false);
}

#[test]
fn cpu_ball_points_publish_authored_state_after_bound_edits() {
    point_publication_cpu(true);
}

fn point_publication_cpu(ball: bool) {
    let (mut model, mut initial, control) = point_fixture(ball);
    let mut next = model.state().unwrap();
    let mut solver = FeatherstoneSolver::new_cpu();
    for step in 0..4 {
        if step == 2 {
            edit_point_targets(&mut model);
        }
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
        verify_points(&model, &next, &[], false);
        std::mem::swap(&mut initial, &mut next);
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_d6_points_publish_authored_state_after_bound_edits() {
    point_publication_gpu(false);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_ball_points_publish_authored_state_after_bound_edits() {
    point_publication_gpu(true);
}

#[cfg(feature = "gpu")]
fn point_publication_gpu(ball: bool) {
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?}");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let (mut model, mut initial, mut control) = point_fixture(ball);
    model.upload_to_device(device, queue).unwrap();
    initial.upload_to_device(device, queue).unwrap();
    control.upload_to_device(device, queue).unwrap();
    let mut next = model.state().unwrap();
    next.upload_to_device(device, queue).unwrap();
    let mut solver = FeatherstoneSolver::new_gpu(&engine, &model).unwrap();
    solver
        .prepare_gpu_coordinate_compensation(&[&initial, &next])
        .unwrap();
    solver.seed_gpu_coordinate_compensation(&initial).unwrap();
    let references = model
        .joint_limit_references
        .device_buffer_current()
        .unwrap()
        .clone();
    for step in 0..4 {
        if step == 2 {
            edit_point_targets(&mut model);
            model.upload_to_device(device, queue).unwrap();
            assert_eq!(
                &references,
                model
                    .joint_limit_references
                    .device_buffer_current()
                    .unwrap()
            );
        }
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
        verify_points(&model, &next, &residues, true);
        std::mem::swap(&mut initial, &mut next);
    }
}

const ROTATION_DT: f64 = 0.00001;

struct RotationCase {
    coordinate: usize,
    initial: Quat,
    rate: Vec3,
}

fn rotation_fixture() -> (Model, State, Control, Vec<RotationCase>) {
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), Vec3::ZERO);
    let mut cases = Vec::new();
    let mut rates = Vec::new();
    for kind in [
        molla_sim::JointType::Ball,
        molla_sim::JointType::Distance,
        molla_sim::JointType::Free,
        molla_sim::JointType::D6,
    ] {
        for frame in [
            Quat::IDENTITY,
            Quat::from_euler(glam::EulerRot::XYZ, 0.2, -0.3, 0.4),
        ] {
            for negative in [false, true] {
                for axis in [Vec3::X, Vec3::Y, Vec3::Z, Vec3::new(1., -2., 3.)] {
                    builder.begin_articulation();
                    let body = builder.add_body(
                        BodyParams::new()
                            .with_mass(2.)
                            .with_inertia(Mat3::IDENTITY * 3.),
                    );
                    let parent = Transform::new(Vec3::new(1.2, -0.4, 0.8), frame);
                    let child = Transform::new(
                        Vec3::ZERO,
                        Quat::from_euler(glam::EulerRot::XYZ, -0.4, 0.3, -0.2),
                    );
                    let joint = match kind {
                        molla_sim::JointType::Ball => {
                            builder.add_joint_ball(BodyId::NONE, body, parent, child)
                        }
                        molla_sim::JointType::Distance => {
                            builder.add_joint_distance(BodyId::NONE, body, parent, child)
                        }
                        molla_sim::JointType::Free => {
                            builder.add_joint_free(BodyId::NONE, body, parent, child)
                        }
                        molla_sim::JointType::D6 => {
                            builder.add_joint_d6(BodyId::NONE, body, parent, child)
                        }
                        _ => unreachable!(),
                    }
                    .unwrap();
                    let count = if kind == molla_sim::JointType::Ball {
                        3
                    } else {
                        6
                    };
                    builder
                        .set_joint_limits(joint, &vec![f64::NAN; count], &vec![f64::NAN; count])
                        .unwrap();
                    cases.push(RotationCase {
                        coordinate: 0,
                        initial: Quat::from_euler(glam::EulerRot::XYZ, 0.3, -0.6, 0.2)
                            * if negative { -1. } else { 1. },
                        rate: axis * 1e-5,
                    });
                    if count == 6 {
                        rates.extend(Vec3::ZERO.to_array());
                    }
                    rates.extend((axis * 1e-5).to_array());
                }
            }
        }
    }
    assert_eq!(cases.len(), 64);
    let model = builder.build().unwrap();
    let mut initial = model.state().unwrap();
    for (joint, case) in cases.iter_mut().enumerate() {
        let offset = if model.joint_type.host().unwrap()[joint] == molla_sim::JointType::Ball as u32
        {
            0
        } else {
            3
        };
        case.coordinate = model.joint_coord_offset.host().unwrap()[joint] as usize + offset;
        initial.joint_q.host_mut().unwrap()[case.coordinate..case.coordinate + 4]
            .copy_from_slice(&case.initial.to_array());
    }
    initial.joint_qd.host_mut().unwrap().copy_from_slice(&rates);
    molla_sim::eval_fk(&model, &mut initial).unwrap();
    let control = model.control();
    (model, initial, control, cases)
}

fn verify_rotations(state: &State, residues: &[f32], cases: &[RotationCase]) {
    for (i, case) in cases.iter().enumerate() {
        let retained = Quat::from_array(std::array::from_fn(|axis| {
            let c = case.coordinate + axis;
            state.joint_q.host().unwrap()[c] + residues.get(c).copied().unwrap_or(0.) as f64
        }))
        .normalize();
        let expected = Quat::from_scaled_axis(case.rate * (ROTATION_DT * 8.)) * case.initial;
        let error = (retained * expected.conjugate()).xyz().length() * 2.;
        println!("quaternion_retained_motion case={i} rotation_error={error:.9e}");
        assert!(error.is_finite() && error <= 1e-12);
    }
}

#[test]
fn cpu_quaternion_kinds_advance_retained_sub_f32_rotation() {
    let (model, mut current, control, cases) = rotation_fixture();
    let mut next = model.state().unwrap();
    let mut solver = FeatherstoneSolver::new_cpu();
    for _ in 0..8 {
        solver
            .step(
                &model,
                &current,
                &control,
                &Contacts::with_capacity(0),
                ROTATION_DT,
                &mut next,
            )
            .unwrap();
        std::mem::swap(&mut current, &mut next);
    }
    verify_rotations(&current, &[], &cases);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_quaternion_kinds_advance_retained_sub_f32_rotation() {
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?} dt={ROTATION_DT} steps=8 cases=64");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let (mut model, mut current, mut control, cases) = rotation_fixture();
    model.upload_to_device(device, queue).unwrap();
    control.upload_to_device(device, queue).unwrap();
    current.upload_to_device(device, queue).unwrap();
    let mut next = model.state().unwrap();
    next.upload_to_device(device, queue).unwrap();
    let mut solver = FeatherstoneSolver::new_gpu(&engine, &model).unwrap();
    solver
        .prepare_gpu_coordinate_compensation(&[&current, &next])
        .unwrap();
    solver.seed_gpu_coordinate_compensation(&current).unwrap();
    for _ in 0..8 {
        solver
            .step(
                &model,
                &current,
                &control,
                &Contacts::with_capacity(0),
                ROTATION_DT,
                &mut next,
            )
            .unwrap();
        assert_eq!(
            gpu_readback::read::<u32>(&engine, solver.gpu_status_buffer().unwrap(), 1)[0],
            0
        );
        std::mem::swap(&mut current, &mut next);
    }
    current.joint_q.download(device, queue).unwrap();
    let residues = gpu_readback::read::<f32>(
        &engine,
        solver.gpu_coordinate_compensation_buffer(&current).unwrap(),
        model.joint_coord_total,
    );
    verify_rotations(&current, &residues, &cases);
}
