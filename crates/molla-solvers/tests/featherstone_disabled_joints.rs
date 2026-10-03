use molla_core::{BodyId, WorldId};
use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::{BodyParams, Contacts, Control, Model, ModelBuilder, State};
use molla_solvers::{FeatherstoneSolver, Solver};

#[cfg(feature = "gpu")]
#[path = "common/gpu_readback.rs"]
mod gpu_readback;

const DT: f64 = 0.01;

struct Case {
    parent: BodyId,
    child: BodyId,
    coordinates: std::ops::Range<usize>,
    rates: std::ops::Range<usize>,
    floating: bool,
}

fn fixture() -> (Model, State, Control, Vec<Case>) {
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), Vec3::ZERO);
    let mut cases = Vec::new();
    for floating in [false, true] {
        for kind in 0..6 {
            builder.begin_articulation();
            let params = BodyParams::new().with_mass(1.).with_inertia(Mat3::IDENTITY);
            let parent = builder.add_body(params.clone());
            let child = builder.add_body(params);
            if floating {
                builder
                    .add_joint_free(
                        BodyId::NONE,
                        parent,
                        Transform::IDENTITY,
                        Transform::IDENTITY,
                    )
                    .unwrap();
            } else {
                builder
                    .add_joint_fixed(
                        BodyId::NONE,
                        parent,
                        Transform::IDENTITY,
                        Transform::IDENTITY,
                    )
                    .unwrap();
            }
            let joint = match kind {
                0 => builder.add_joint_revolute(
                    parent,
                    child,
                    Transform::IDENTITY,
                    Transform::IDENTITY,
                    Vec3::X,
                ),
                1 => builder.add_joint_prismatic(
                    parent,
                    child,
                    Transform::IDENTITY,
                    Transform::IDENTITY,
                    Vec3::Y,
                ),
                2 => {
                    builder.add_joint_ball(parent, child, Transform::IDENTITY, Transform::IDENTITY)
                }
                3 => {
                    builder.add_joint_free(parent, child, Transform::IDENTITY, Transform::IDENTITY)
                }
                4 => builder.add_joint_distance(
                    parent,
                    child,
                    Transform::IDENTITY,
                    Transform::IDENTITY,
                ),
                _ => builder.add_joint_d6(parent, child, Transform::IDENTITY, Transform::IDENTITY),
            }
            .unwrap();
            let width = if kind < 2 {
                1
            } else if kind == 2 {
                3
            } else {
                6
            };
            let mut lower = vec![-0.001; width];
            if kind == 4 {
                lower[0] = 0.;
            }
            let mut upper = vec![0.001; width];
            if kind == 5 {
                lower[3] = 0.;
                upper[3] = 0.;
            }
            builder.set_joint_limits(joint, &lower, &upper).unwrap();
            builder
                .set_joint_velocity_limit(joint, &vec![0.0001; width])
                .unwrap();
            builder.set_joint_enabled(joint, false).unwrap();
            cases.push((parent, child, joint, floating));
        }
    }
    let model = builder.build().unwrap();
    let mut state = model.state().unwrap();
    let mut control = model.control();
    let cases: Vec<_> = cases
        .into_iter()
        .map(|(parent, child, joint, floating)| {
            let index = joint.index();
            let coordinates = model.joint_coord_offset.host().unwrap()[index] as usize
                ..model.joint_coord_offset.host().unwrap()[index + 1] as usize;
            let rates = model.joint_dof_offset.host().unwrap()[index] as usize
                ..model.joint_dof_offset.host().unwrap()[index + 1] as usize;
            state.joint_qd.host_mut().unwrap()[rates.clone()].fill(100.);
            control.joint_force.host_mut().unwrap()[rates.clone()].fill(10.);
            state.body_f.host_mut().unwrap()[child.index()] = SpatialVector::new(Vec3::Y, Vec3::X);
            Case {
                parent,
                child,
                coordinates,
                rates,
                floating,
            }
        })
        .collect();
    molla_sim::eval_fk(&model, &mut state).unwrap();
    molla_sim::eval_body_velocities(&model, &mut state).unwrap();
    (model, state, control, cases)
}

fn verify(initial: &State, state: &State, cases: &[Case], step: usize, tolerance: f64) {
    for (i, case) in cases.iter().enumerate() {
        let speed = if case.floating { DT / 2. } else { 0. };
        let expected = SpatialVector::new(Vec3::Y * speed, Vec3::X * speed);
        let expected_rotation = Quat::from_rotation_x(speed * DT * step as f64);
        for body in [case.parent, case.child] {
            let pose = state.body_q.host().unwrap()[body.index()];
            let velocity = state.body_qd.host().unwrap()[body.index()];
            assert!(
                (velocity.linear - expected.linear).length() <= tolerance,
                "case={i} step={step} linear={velocity:?}"
            );
            assert!(
                (velocity.angular - expected.angular).length() <= tolerance,
                "case={i} step={step} angular={velocity:?}"
            );
            assert!((pose.position - expected.linear * DT * step as f64).length() <= tolerance);
            assert!(
                2. * (pose.rotation.normalize() * expected_rotation.conjugate())
                    .xyz()
                    .length()
                    <= tolerance
            );
        }
        for coordinate in case.coordinates.clone() {
            assert!(
                (state.joint_q.host().unwrap()[coordinate]
                    - initial.joint_q.host().unwrap()[coordinate])
                    .abs()
                    <= tolerance
            );
        }
        assert!(
            state.joint_qd.host().unwrap()[case.rates.clone()]
                .iter()
                .all(|&v| v == 0.)
        );
    }
}

#[test]
fn cpu_disabled_joints_freeze_relative_motion_and_transmit_child_loads() {
    let (model, initial, control, cases) = fixture();
    let mut next = model.state().unwrap();
    let mut solver = FeatherstoneSolver::new_cpu();
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
    verify(&initial, &next, &cases, 1, 1e-8);
    let source = next.clone_host().unwrap();
    solver.drift_cpu(&model, &source, DT, &mut next).unwrap();
    verify(&initial, &next, &cases, 2, 1e-8);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_disabled_joints_freeze_relative_motion_and_transmit_child_loads() {
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?}");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let (mut model, initial, mut control, cases) = fixture();
    model.upload_to_device(device, queue).unwrap();
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
    verify(&initial, &b, &cases, 1, 1e-3);
    std::mem::swap(&mut a, &mut b);
    b = a.clone_host().unwrap();
    b.upload_to_device(device, queue).unwrap();
    solver.drift_gpu(&model, &a, &control, DT, &mut b).unwrap();
    b.download_from_device(device, queue).unwrap();
    b.joint_q.download(device, queue).unwrap();
    b.joint_qd.download(device, queue).unwrap();
    assert_eq!(
        gpu_readback::read::<u32>(&engine, solver.gpu_status_buffer().unwrap(), 1)[0],
        0
    );
    verify(&initial, &b, &cases, 2, 1e-3);
}
