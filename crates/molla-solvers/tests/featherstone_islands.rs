use molla_core::BodyId;
use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::{
    BodyFlags, BodyParams, Contacts, Control, JointTargetMode, Model, ModelBuilder, State,
};
use molla_solvers::{FeatherstoneSolver, Solver};

fn fixture(ids: &[usize]) -> (Model, State, Control) {
    fixture_with_mimic(ids, false)
}

fn fixture_with_mimic(ids: &[usize], mimic: bool) -> (Model, State, Control) {
    let mut builder = ModelBuilder::new();
    for &id in ids {
        builder.begin_articulation();
        let pose = Transform::from_translation(Vec3::new(id as f64 * 3.0, 2.0, -0.3));
        let base = builder.add_body(
            BodyParams::new()
                .with_mass(4.0)
                .with_inertia(Mat3::IDENTITY * 2.0)
                .with_pose(pose),
        );
        let child = builder.add_body(
            BodyParams::new()
                .with_mass(1.0)
                .with_inertia(Mat3::IDENTITY * 0.3)
                .with_com(Vec3::new(0.1, -0.2, 0.05)),
        );
        builder
            .add_joint_free(BodyId::NONE, base, pose, Transform::IDENTITY)
            .unwrap();
        let hinge = builder
            .add_joint_revolute(
                base,
                child,
                Transform::from_translation(Vec3::new(0.2, -0.4, 0.1)),
                Transform::IDENTITY,
                Vec3::Z,
            )
            .unwrap();
        let (lower, upper) = if id % 4 == 2 { (0.0, 0.0) } else { (-0.2, 0.2) };
        builder.set_joint_limits(hinge, &[lower], &[upper]).unwrap();
    }
    if mimic {
        builder
            .add_mimic_constraint(molla_core::JointId(1), molla_core::JointId(3), 0.1, -1.0)
            .unwrap();
    }
    let mut model = builder.build().unwrap();
    for (local, &id) in ids.iter().enumerate() {
        if id % 5 == 0 {
            model.body_flags.host_mut().unwrap()[2 * local] = BodyFlags::FIXED;
        }
        if id % 4 == 1 {
            model.joint_enabled.host_mut().unwrap()[2 * local + 1] = 0;
        }
    }
    let mut state = model.state().unwrap();
    let mut control = model.control();
    for (local, &id) in ids.iter().enumerate() {
        state.joint_qd.host_mut().unwrap()[7 * local + 4] = 0.03 * id as f64;
        state.body_f.host_mut().unwrap()[2 * local + 1] = SpatialVector::new(
            Vec3::new(id as f64 * 0.2, 0.3, -0.1),
            Vec3::new(0.2, -0.1, 0.3),
        );
        let dof = 7 * local + 6;
        control.joint_target_mode.host_mut().unwrap()[dof] = JointTargetMode::Velocity as u32;
        control.joint_target_vel.host_mut().unwrap()[dof] = if id % 2 == 0 { 2.0 } else { -2.0 };
        control.joint_target_kd.host_mut().unwrap()[dof] = 8.0;
        control.joint_motor_max_force.host_mut().unwrap()[dof] = 20.0;
    }
    molla_sim::eval_fk(&model, &mut state).unwrap();
    molla_sim::eval_body_velocities(&model, &mut state).unwrap();
    (model, state, control)
}

#[path = "featherstone_islands/activation.rs"]
mod activation;

#[test]
fn independent_articulations_match_separate_simulations() {
    for rigorous in [false, true] {
        let ids = [0, 1, 2, 3, 4, 5, 6, 7];
        let (model, mut state, control) = fixture(&ids);
        let mut next = state.clone_host().unwrap();
        let mut separate: Vec<_> = ids
            .iter()
            .map(|&id| {
                let (model, state, control) = fixture(&[id]);
                let next = state.clone_host().unwrap();
                (model, state, control, next)
            })
            .collect();
        let mut solver = FeatherstoneSolver::new_cpu()
            .with_rigorous_d_solve(rigorous)
            .with_stable_motor_feedback(true)
            .with_substeps(2);
        let contacts = Contacts::with_capacity(0);
        for step in 0..120 {
            solver
                .step(&model, &state, &control, &contacts, 1.0 / 120.0, &mut next)
                .unwrap();
            std::mem::swap(&mut state, &mut next);
            for (local, (model, isolated, control, next)) in separate.iter_mut().enumerate() {
                solver
                    .step(model, isolated, control, &contacts, 1.0 / 120.0, next)
                    .unwrap();
                std::mem::swap(isolated, next);
                assert_eq!(
                    &state.joint_q.host().unwrap()[8 * local..8 * local + 8],
                    isolated.joint_q.host().unwrap(),
                    "q: rigorous={rigorous}, step={step}, island={local}"
                );
                assert_eq!(
                    &state.joint_qd.host().unwrap()[7 * local..7 * local + 7],
                    isolated.joint_qd.host().unwrap(),
                    "qd: rigorous={rigorous}, step={step}, island={local}"
                );
                assert_eq!(
                    &state.body_q.host().unwrap()[2 * local..2 * local + 2],
                    isolated.body_q.host().unwrap()
                );
                assert_eq!(
                    &state.body_qd.host().unwrap()[2 * local..2 * local + 2],
                    isolated.body_qd.host().unwrap()
                );
            }
        }
        let q = state.joint_q.host().unwrap();
        assert!(q[8 * 3 + 7] < -0.19 && q[8 * 4 + 7] > 0.19);
        assert_eq!(q[8 + 7], 0.0);
        assert_eq!(q[8 * 2 + 7], 0.0);
        assert!(state.body_q.host().unwrap()[8].position.y < 1.0);
    }
}

#[test]
#[ignore = "CPU scaling benchmark; run in release without concurrent builds"]
fn independent_articulation_scaling() {
    for count in [1, 32, 128, 512] {
        let (model, state, control) = fixture(&(0..count).collect::<Vec<_>>());
        let mut next = state.clone_host().unwrap();
        let mut solver = FeatherstoneSolver::new_cpu()
            .with_rigorous_d_solve(true)
            .with_stable_motor_feedback(true);
        let contacts = Contacts::with_capacity(0);
        let mut samples = Vec::new();
        for sample in 0..100 {
            let start = std::time::Instant::now();
            solver
                .step(&model, &state, &control, &contacts, 1.0 / 120.0, &mut next)
                .unwrap();
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            assert!(next.joint_q.host().unwrap().iter().all(|v| v.is_finite()));
            assert!(next.joint_qd.host().unwrap().iter().all(|v| v.is_finite()));
            if sample >= 20 {
                samples.push(elapsed);
            }
        }
        samples.sort_by(f64::total_cmp);
        eprintln!(
            "articulation scaling: articulations={count} bodies={} dofs={} median_ms={:.6} p95_ms={:.6} samples={}",
            model.body_count,
            model.joint_dof_total,
            samples[40],
            samples[76],
            samples.len()
        );
    }
}
