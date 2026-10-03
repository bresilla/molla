use molla_core::{BodyId, WorldId};
use molla_math::{Mat3, Quat, SpatialInertia, SpatialVector, Transform, Vec3};
use molla_sim::{BodyParams, Contacts, Control, Model, ModelBuilder, State};
use molla_solvers::{FeatherstoneSolver, Solver};

#[cfg(feature = "gpu")]
#[path = "common/gpu_readback.rs"]
mod gpu_readback;

const DT: f64 = 0.01;

fn fixture() -> (Model, State) {
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), Vec3::ZERO);
    for _ in 0..2 {
        builder.begin_articulation();
        let params = BodyParams::new().with_mass(1.).with_inertia(Mat3::IDENTITY);
        let root = builder.add_body(params.clone());
        let tip = builder.add_body(params);
        let d6 = builder
            .add_joint_d6(BodyId::NONE, root, Transform::IDENTITY, Transform::IDENTITY)
            .unwrap();
        builder
            .set_joint_limits(
                d6,
                &[0., 0., 0., 0., f64::NEG_INFINITY, f64::NEG_INFINITY],
                &[0., 0., 0., 0., f64::INFINITY, f64::INFINITY],
            )
            .unwrap();
        let distance = builder
            .add_joint_distance(root, tip, Transform::IDENTITY, Transform::IDENTITY)
            .unwrap();
        builder
            .set_joint_limits(
                distance,
                &[f64::NAN; 6],
                &[1., f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN],
            )
            .unwrap();
    }
    let model = builder.build().unwrap();
    let mut initial = model.state().unwrap();
    for (case, radius) in [0.5, 1.].into_iter().enumerate() {
        let coord = model.joint_coord_offset.host().unwrap()[2 * case + 1] as usize;
        initial.joint_q.host_mut().unwrap()[coord] = radius;
        initial.body_f.host_mut().unwrap()[2 * case + 1] = SpatialVector::new(Vec3::X, Vec3::ZERO);
    }
    molla_sim::eval_fk(&model, &mut initial).unwrap();
    (model, initial)
}

fn verify(state: &State, tolerance: f64, step: usize) {
    for case in 0..2 {
        let speed = if case == 0 { DT } else { 0. };
        let radius = if case == 0 {
            0.5 + speed * DT * step as f64
        } else {
            1.
        };
        let tip = 2 * case + 1;
        let velocity = state.body_qd.host().unwrap()[tip];
        let pose = state.body_q.host().unwrap()[tip];
        assert!(
            (velocity.linear - Vec3::X * speed).length() <= tolerance,
            "case={case} step={step} velocity={velocity:?}"
        );
        assert!(velocity.angular.length() <= tolerance);
        assert!(
            (pose.position - Vec3::X * radius).length() <= tolerance,
            "case={case} step={step} pose={pose:?}"
        );
        assert!(state.body_q.host().unwrap()[tip - 1].position.length() <= tolerance);
        assert_eq!(state.body_qd.host().unwrap()[tip - 1], SpatialVector::ZERO);
    }
}

#[test]
fn cpu_mixed_stops_keep_slack_distance_free_and_react_at_maximum() {
    let (model, initial) = fixture();
    let control = model.control();
    let mut solver = FeatherstoneSolver::new_cpu();
    let mut next = model.state().unwrap();
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
    verify(&next, 1e-8, 1);
    let source = next.clone_host().unwrap();
    solver.drift_cpu(&model, &source, DT, &mut next).unwrap();
    verify(&next, 1e-8, 2);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_mixed_stops_keep_slack_distance_free_and_react_at_maximum() {
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?}");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let (mut model, initial) = fixture();
    model.upload_to_device(device, queue).unwrap();
    let mut control = model.control();
    control.upload_to_device(device, queue).unwrap();
    let mut a = initial.clone_host().unwrap();
    let mut b = model.state().unwrap();
    a.upload_to_device(device, queue).unwrap();
    b.upload_to_device(device, queue).unwrap();
    let mut solver = FeatherstoneSolver::new_gpu(&engine, &model).unwrap();
    solver
        .step(
            &model,
            &a,
            &control,
            &Contacts::with_capacity(0),
            DT,
            &mut b,
        )
        .unwrap();
    b.download_from_device(device, queue).unwrap();
    b.joint_q.download(device, queue).unwrap();
    b.joint_qd.download(device, queue).unwrap();
    assert_eq!(
        gpu_readback::read::<u32>(&engine, solver.gpu_status_buffer().unwrap(), 1)[0],
        0
    );
    verify(&b, 1e-3, 1);
    std::mem::swap(&mut a, &mut b);
    b = a.clone_host().unwrap();
    b.upload_to_device(device, queue).unwrap();
    solver.drift_gpu(&model, &a, &control, DT, &mut b).unwrap();
    b.download_from_device(device, queue).unwrap();
    assert_eq!(
        gpu_readback::read::<u32>(&engine, solver.gpu_status_buffer().unwrap(), 1)[0],
        0
    );
    verify(&b, 1e-3, 2);
}

const RADIAL_DT: f64 = 0.00001;

struct RadialCase {
    body: usize,
    parent: Option<usize>,
    coordinate: usize,
    dof: usize,
    lower: f64,
    upper: f64,
    initial_position: Vec3,
    initial_rotation: Quat,
    frame_rotation: Quat,
    anchor_acceleration: Vec3,
    com_acceleration: Vec3,
    angular_acceleration: Vec3,
}

fn radial_fixture(standalone: bool) -> (Model, State, Control, Vec<RadialCase>) {
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
    let frame_parent = Transform::new(
        Vec3::new(0.1, 0.2, -0.3),
        Quat::from_euler(glam::EulerRot::XYZ, -0.3, 0.2, 0.4),
    );
    let frame_child = Transform::new(
        Vec3::new(-0.3, 0.5, -0.9),
        Quat::from_euler(glam::EulerRot::XYZ, -0.4, 0.3, -0.2),
    );
    let mut cases = Vec::new();
    let mut loads = Vec::new();
    for frame in [
        Quat::IDENTITY,
        Quat::from_euler(glam::EulerRot::XYZ, 0.2, -0.4, 0.3),
    ] {
        let root_pose = Transform::new(Vec3::new(1.2, -0.4, 0.8), frame);
        let anchor_frame = root_pose.compose(&frame_parent);
        for radial_axis in [Vec3::X, Vec3::new(1., 2., -1.).normalize()] {
            for point in [false, true] {
                for upper_stop in [false, true] {
                    for middle in [false, true] {
                        if point && middle {
                            continue;
                        }
                        for outward in [false, true] {
                            builder.begin_articulation();
                            let root = if standalone {
                                BodyId::NONE
                            } else {
                                builder.add_body(
                                    BodyParams::new()
                                        .with_mass(3.)
                                        .with_inertia(Mat3::IDENTITY * 4.),
                                )
                            };
                            let child =
                                builder.add_body(BodyParams::new().with_spatial_inertia(inertia));
                            if !standalone {
                                let d6 = builder
                                    .add_joint_d6(
                                        BodyId::NONE,
                                        root,
                                        root_pose,
                                        Transform::IDENTITY,
                                    )
                                    .unwrap();
                                builder
                                    .set_joint_limits(
                                        d6,
                                        &[
                                            0.,
                                            0.,
                                            0.,
                                            f64::NEG_INFINITY,
                                            f64::NEG_INFINITY,
                                            f64::NEG_INFINITY,
                                        ],
                                        &[0., 0., 0., f64::INFINITY, f64::INFINITY, f64::INFINITY],
                                    )
                                    .unwrap();
                                builder
                                    .set_joint_velocity_limit(
                                        d6,
                                        &[f64::INFINITY, f64::INFINITY, f64::INFINITY, 0., 0., 0.],
                                    )
                                    .unwrap();
                            }
                            let distance = builder
                                .add_joint_distance(
                                    root,
                                    child,
                                    if standalone {
                                        anchor_frame
                                    } else {
                                        frame_parent
                                    },
                                    frame_child,
                                )
                                .unwrap();
                            let target = 1.5 + 1e-9;
                            let (lower, upper) = if point {
                                (target, target)
                            } else if upper_stop {
                                (target - 1e-8, target)
                            } else {
                                (target, target + 1e-8)
                            };
                            assert_eq!(lower as f32, upper as f32);
                            builder
                                .set_joint_limits(
                                    distance,
                                    &[lower, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN],
                                    &[upper, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN],
                                )
                                .unwrap();
                            let radius = if middle {
                                0.5 * lower + 0.5 * upper
                            } else if upper_stop {
                                upper
                            } else {
                                lower
                            };
                            let initial_position = radial_axis * radius;
                            let initial_rotation =
                                Quat::from_euler(glam::EulerRot::XYZ, 0.3, -0.6, 0.2)
                                    * if upper_stop { -1. } else { 1. };
                            let child_pose = anchor_frame
                                .compose(&Transform::new(initial_position, initial_rotation))
                                .compose(&frame_child.inverse());
                            let arm = child_pose.rotation * (inertia.com - frame_child.position);
                            let rotation = Mat3::from_quat(child_pose.rotation);
                            let inverse_inertia =
                                (rotation * inertia.inertia * rotation.transpose()).inverse();
                            let normal = anchor_frame.rotation * radial_axis;
                            let direction =
                                if upper_stop { 1. } else { -1. } * if outward { 1. } else { -1. };
                            let free_acceleration = normal * direction;
                            let constrained = point || (!middle && outward);
                            let moment = arm.cross(normal);
                            let reaction = if constrained {
                                -normal.dot(free_acceleration)
                                    / (1. / inertia.mass + moment.dot(inverse_inertia * moment))
                            } else {
                                0.
                            };
                            let angular_acceleration = -(inverse_inertia * moment) * reaction;
                            let com_acceleration =
                                free_acceleration + normal * (reaction / inertia.mass);
                            let anchor_acceleration =
                                com_acceleration - angular_acceleration.cross(arm);
                            cases.push(RadialCase {
                                body: child.index(),
                                parent: (!standalone).then(|| root.index()),
                                coordinate: 0,
                                dof: 0,
                                lower,
                                upper,
                                initial_position,
                                initial_rotation,
                                frame_rotation: anchor_frame.rotation,
                                anchor_acceleration,
                                com_acceleration,
                                angular_acceleration,
                            });
                            loads.push(SpatialVector::new(
                                normal * (inertia.mass * direction),
                                Vec3::ZERO,
                            ));
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases.len(), 48);
    let model = builder.build().unwrap();
    let mut initial = model.state().unwrap();
    for (i, case) in cases.iter_mut().enumerate() {
        let joint = if standalone { i } else { 2 * i + 1 };
        case.coordinate = model.joint_coord_offset.host().unwrap()[joint] as usize;
        case.dof = model.joint_dof_offset.host().unwrap()[joint] as usize;
        let c = case.coordinate;
        initial.joint_q.host_mut().unwrap()[c..c + 3]
            .copy_from_slice(&case.initial_position.to_array());
        initial.joint_q.host_mut().unwrap()[c + 3..c + 7]
            .copy_from_slice(&case.initial_rotation.to_array());
        initial.body_f.host_mut().unwrap()[case.body] = loads[i];
    }
    molla_sim::eval_fk(&model, &mut initial).unwrap();
    let control = model.control();
    (model, initial, control, cases)
}

fn verify_radial_precision(next: &State, residues: &[f32], cases: &[RadialCase]) {
    for (i, case) in cases.iter().enumerate() {
        let c = case.coordinate;
        let component = |axis: usize| {
            next.joint_q.host().unwrap()[c + axis]
                + residues.get(c + axis).copied().unwrap_or(0.) as f64
        };
        let position = Vec3::new(component(0), component(1), component(2));
        let rotation =
            Quat::from_xyzw(component(3), component(4), component(5), component(6)).normalize();
        let rate = next.body_qd.host().unwrap()[case.body];
        let acceleration_error = (rate.linear / RADIAL_DT - case.com_acceleration).length();
        let angular_error = (rate.angular / RADIAL_DT - case.angular_acceleration).length();
        let expected_position = case.initial_position
            + case.frame_rotation.conjugate() * case.anchor_acceleration * (RADIAL_DT * RADIAL_DT);
        let expected_rotation = Quat::from_scaled_axis(
            case.frame_rotation.conjugate() * case.angular_acceleration * (RADIAL_DT * RADIAL_DT),
        ) * case.initial_rotation;
        let position_error = (position - expected_position).length();
        let rotation_error = (rotation * expected_rotation.conjugate()).xyz().length() * 2.;
        let radius = position.length();
        let violation = (case.lower - radius).max(radius - case.upper).max(0.);
        println!(
            "mixed_radial_precision case={i} acceleration_error={acceleration_error:.9e} angular_error={angular_error:.9e} position_error={position_error:.9e} rotation_error={rotation_error:.9e} violation={violation:.9e}"
        );
        assert!(acceleration_error <= 0.01 * case.com_acceleration.length().max(1.));
        assert!(angular_error <= 0.01 * case.angular_acceleration.length().max(1.));
        assert!(position_error <= 1e-12 && rotation_error <= 1e-12 && violation <= 1e-12);
        if let Some(parent) = case.parent {
            let root_rate = next.body_qd.host().unwrap()[parent];
            assert!(root_rate.linear.length() <= 1e-9 && root_rate.angular.length() <= 1e-9);
        }
    }
}

#[test]
fn cpu_mixed_radial_intervals_preserve_authored_bounds_and_physical_response() {
    radial_precision_cpu(false);
}

#[test]
fn cpu_standalone_radial_intervals_use_coupled_anchor_mass() {
    radial_precision_cpu(true);
}

fn radial_precision_cpu(standalone: bool) {
    let (model, initial, control, cases) = radial_fixture(standalone);
    let mut next = model.state().unwrap();
    FeatherstoneSolver::new_cpu()
        .step(
            &model,
            &initial,
            &control,
            &Contacts::with_capacity(0),
            RADIAL_DT,
            &mut next,
        )
        .unwrap();
    verify_radial_precision(&next, &[], &cases);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_mixed_radial_intervals_preserve_authored_bounds_and_physical_response() {
    radial_precision_gpu(false);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_standalone_radial_intervals_use_coupled_anchor_mass() {
    radial_precision_gpu(true);
}

#[cfg(feature = "gpu")]
fn radial_precision_gpu(standalone: bool) {
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?} dt={RADIAL_DT} width=1e-8 cases=48");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let (mut model, mut initial, mut control, cases) = radial_fixture(standalone);
    model.upload_to_device(device, queue).unwrap();
    control.upload_to_device(device, queue).unwrap();
    let mut next = model.state().unwrap();
    initial.upload_to_device(device, queue).unwrap();
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
            RADIAL_DT,
            &mut next,
        )
        .unwrap();
    next.download_from_device(device, queue).unwrap();
    next.joint_q.download(device, queue).unwrap();
    next.joint_qd.download(device, queue).unwrap();
    assert_eq!(
        gpu_readback::read::<u32>(&engine, solver.gpu_status_buffer().unwrap(), 1)[0],
        0
    );
    let residues = gpu_readback::read::<f32>(
        &engine,
        solver.gpu_coordinate_compensation_buffer(&next).unwrap(),
        model.joint_coord_total,
    );
    verify_radial_precision(&next, &residues, &cases);
}

struct SemanticCase {
    coordinate: usize,
    position: Vec3,
    velocity: Vec3,
    rotation: Quat,
}

fn semantic_fixture() -> (Model, State, Control, Vec<SemanticCase>) {
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), Vec3::ZERO);
    let mut cases = Vec::new();
    let mut velocities = Vec::new();
    for frame in [
        Quat::IDENTITY,
        Quat::from_euler(glam::EulerRot::XYZ, 0.2, -0.3, 0.4),
    ] {
        for (lo, hi, radius) in [
            (f64::NAN, f64::NAN, 1.5),
            (1., f64::NAN, 1.5),
            (f64::NAN, 2., 1.5),
            (1., 2., 1.5),
            (1., f64::NAN, 1.),
            (f64::NAN, 2., 2.),
            (1.5, 1.5, 1.5),
            (0., 0., 0.),
        ] {
            for velocity in [-0.25, 0.25] {
                builder.begin_articulation();
                let body =
                    builder.add_body(BodyParams::new().with_mass(2.).with_inertia(Mat3::IDENTITY));
                let joint = builder
                    .add_joint_distance(
                        BodyId::NONE,
                        body,
                        Transform::new(Vec3::new(1.2, -0.4, 0.8), frame),
                        Transform::IDENTITY,
                    )
                    .unwrap();
                builder
                    .set_joint_limits(
                        joint,
                        &[lo, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN],
                        &[hi, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN],
                    )
                    .unwrap();
                let axis = Vec3::new(1., 2., -1.).normalize();
                let blocked = (lo.is_finite() && radius <= lo && velocity < 0.)
                    || (hi.is_finite() && radius >= hi && velocity > 0.);
                let rate = if blocked { Vec3::ZERO } else { axis * velocity };
                cases.push(SemanticCase {
                    coordinate: 0,
                    position: axis * radius + rate * RADIAL_DT,
                    velocity: rate,
                    rotation: Quat::from_euler(glam::EulerRot::XYZ, 0.3, -0.6, 0.2)
                        * if velocity > 0. { -1. } else { 1. },
                });
                velocities.push(axis * velocity);
            }
        }
    }
    assert_eq!(cases.len(), 32);
    let model = builder.build().unwrap();
    let mut initial = model.state().unwrap();
    for (j, case) in cases.iter_mut().enumerate() {
        let c = model.joint_coord_offset.host().unwrap()[j] as usize;
        let d = model.joint_dof_offset.host().unwrap()[j] as usize;
        case.coordinate = c;
        let position = case.position - case.velocity * RADIAL_DT;
        initial.joint_q.host_mut().unwrap()[c..c + 3].copy_from_slice(&position.to_array());
        initial.joint_q.host_mut().unwrap()[c + 3..c + 7]
            .copy_from_slice(&case.rotation.to_array());
        initial.joint_qd.host_mut().unwrap()[d..d + 3].copy_from_slice(&velocities[j].to_array());
        initial.joint_qd.host_mut().unwrap()[d + 3..d + 6]
            .copy_from_slice(&Vec3::new(1e-5, -2e-5, 3e-5).to_array());
    }
    molla_sim::eval_fk(&model, &mut initial).unwrap();
    let control = model.control();
    (model, initial, control, cases)
}

fn verify_semantics(model: &Model, next: &State, residues: &[f32], cases: &[SemanticCase]) {
    let omega = Vec3::new(1e-5, -2e-5, 3e-5);
    for (j, case) in cases.iter().enumerate() {
        let c = case.coordinate;
        let d = model.joint_dof_offset.host().unwrap()[j] as usize;
        let value = |axis: usize| {
            next.joint_q.host().unwrap()[c + axis]
                + residues.get(c + axis).copied().unwrap_or(0.) as f64
        };
        let position = Vec3::new(value(0), value(1), value(2));
        let rotation = Quat::from_xyzw(value(3), value(4), value(5), value(6)).normalize();
        let expected = Quat::from_scaled_axis(omega * RADIAL_DT) * case.rotation;
        let position_error = (position - case.position).length();
        let rotation_error = (rotation * expected.conjugate()).xyz().length() * 2.;
        let velocity = Vec3::from_slice(&next.joint_qd.host().unwrap()[d..d + 3]);
        let angular = Vec3::from_slice(&next.joint_qd.host().unwrap()[d + 3..d + 6]);
        println!(
            "distance_semantics case={j} position_error={position_error:.9e} rotation_error={rotation_error:.9e} velocity_error={:.9e}",
            (velocity - case.velocity).length()
        );
        assert!(position_error <= 1e-12 && rotation_error <= 1e-12);
        assert!((velocity - case.velocity).length() <= 1e-7);
        assert!((angular - omega).length() <= 1e-8);
    }
}

#[test]
fn cpu_standalone_distance_slack_unilateral_and_zero_bounds_preserve_rotation() {
    let (model, initial, control, cases) = semantic_fixture();
    let mut next = model.state().unwrap();
    FeatherstoneSolver::new_cpu()
        .step(
            &model,
            &initial,
            &control,
            &Contacts::with_capacity(0),
            RADIAL_DT,
            &mut next,
        )
        .unwrap();
    verify_semantics(&model, &next, &[], &cases);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_standalone_distance_slack_unilateral_and_zero_bounds_preserve_rotation() {
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?} dt={RADIAL_DT} cases=32");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let (mut model, mut initial, mut control, cases) = semantic_fixture();
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
            RADIAL_DT,
            &mut next,
        )
        .unwrap();
    next.joint_q.download(device, queue).unwrap();
    next.joint_qd.download(device, queue).unwrap();
    assert_eq!(
        gpu_readback::read::<u32>(&engine, solver.gpu_status_buffer().unwrap(), 1)[0],
        0
    );
    let residues = gpu_readback::read::<f32>(
        &engine,
        solver.gpu_coordinate_compensation_buffer(&next).unwrap(),
        model.joint_coord_total,
    );
    verify_semantics(&model, &next, &residues, &cases);
}
