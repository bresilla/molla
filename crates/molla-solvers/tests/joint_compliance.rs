use molla_core::{BodyId, JointId};
use molla_math::{Mat3, Quat, SpatialInertia, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::JointSoftness;
use molla_sim::{BodyParams, Contacts, Model, ModelBuilder, State};
use molla_solvers::{FeatherstoneSolver, Solver};

fn softness(frequency: f64, damping: f64) -> JointSoftness {
    JointSoftness {
        natural_frequency: frequency,
        damping_ratio: damping,
    }
}

fn body(builder: &mut ModelBuilder, mass: f64) -> BodyId {
    builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass,
        com: Vec3::ZERO,
        inertia: Mat3::IDENTITY * mass,
    }))
}

fn model(mass: f64, frequency: f64, damping: f64) -> Model {
    let mut builder = ModelBuilder::new();
    let b = body(&mut builder, mass);
    let j = builder
        .add_joint_d6(BodyId::NONE, b, Transform::IDENTITY, Transform::IDENTITY)
        .unwrap();
    builder
        .set_joint_compliance(j, softness(frequency, damping), &[0.0; 6], &[0.0; 6])
        .unwrap();
    let mut model = builder.build().unwrap();
    model.gravity.host_mut().unwrap().fill(Vec3::ZERO);
    model
}

fn advance(model: &Model, state: &mut State, dt: f64, count: usize, substeps: usize) {
    let mut solver = FeatherstoneSolver::new_cpu();
    solver.rigorous_d_solve = true;
    solver.substeps = substeps;
    let mut out = model.state().unwrap();
    let control = model.control();
    for _ in 0..count {
        solver
            .step(
                model,
                state,
                &control,
                &Contacts::with_capacity(0),
                dt,
                &mut out,
            )
            .unwrap();
        std::mem::swap(state, &mut out);
    }
}

fn near(a: f64, b: f64, tolerance: f64) {
    assert!((a - b).abs() < tolerance, "{a} != {b}");
}

#[test]
fn compliance_is_a_force_not_a_position_projection() {
    for mass in [0.01, 1.0, 5000.0] {
        let model = model(mass, 8.0, 1.0);
        let mut state = model.state().unwrap();
        state.joint_q.host_mut().unwrap()[0] = 0.2;
        let dt = 1.0 / 960.0;
        let omega = std::f64::consts::TAU * 8.0;
        let expected_velocity = -omega * omega * 0.2 * dt / (1.0 + omega * dt).powi(2);
        advance(&model, &mut state, dt, 1, 1);
        near(state.joint_qd.host().unwrap()[0], expected_velocity, 1e-10);
        near(
            state.joint_q.host().unwrap()[0],
            0.2 + expected_velocity * dt,
            1e-10,
        );
        assert!(
            state
                .body_f
                .host()
                .unwrap()
                .iter()
                .all(|f| *f == SpatialVector::ZERO)
        );
        assert!(
            model
                .joint_limit_lower
                .host()
                .unwrap()
                .iter()
                .all(|x| *x == f64::NEG_INFINITY)
        );
    }
}

#[test]
fn critically_damped_response_converges_for_hitch_frequencies() {
    for frequency in [8.0, 30.0, 1000.0] {
        let model = model(2700.0, frequency, 1.0);
        let mut state = model.state().unwrap();
        state.joint_q.host_mut().unwrap()[0] = 0.2;
        state.joint_q.host_mut().unwrap()[3..7]
            .copy_from_slice(&Quat::from_rotation_z(0.3).to_array());
        advance(&model, &mut state, 1.0 / 120.0, 120, 8);
        assert!(
            state.joint_q.host().unwrap()[..3]
                .iter()
                .all(|q| q.abs() < 1e-7)
        );
        assert!(
            state
                .joint_qd
                .host()
                .unwrap()
                .iter()
                .all(|q| q.abs() < 1e-6)
        );
        assert!(
            state.body_q.host().unwrap()[0]
                .rotation
                .angle_between(Quat::IDENTITY)
                < 1e-7
        );
    }
}

#[test]
fn zero_frequency_releases_axes_and_does_not_damp() {
    let model = model(3.0, 0.0, 20.0);
    let mut state = model.state().unwrap();
    state.joint_q.host_mut().unwrap()[0] = 0.3;
    state.joint_qd.host_mut().unwrap()[0] = 2.0;
    advance(&model, &mut state, 0.1, 10, 8);
    near(state.joint_q.host().unwrap()[0], 2.3, 1e-10);
    near(state.joint_qd.host().unwrap()[0], 2.0, 1e-10);
}

#[test]
fn external_load_produces_mass_invariant_compliant_deflection() {
    for mass in [1.0, 5000.0] {
        let mut model = model(mass, 8.0, 1.0);
        model.gravity.host_mut().unwrap().fill(-Vec3::Y * 9.81);
        let mut state = model.state().unwrap();
        advance(&model, &mut state, 1.0 / 120.0, 240, 8);
        let omega = std::f64::consts::TAU * 8.0;
        let dt = 1.0 / 960.0;
        let expected = -9.81 * (1.0 + omega * dt).powi(2) / omega.powi(2);
        near(state.joint_q.host().unwrap()[1], expected, 1e-9);
        near(state.joint_qd.host().unwrap()[1], 0.0, 1e-9);
    }
}

#[test]
fn spring_interval_is_unilateral_and_free_inside() {
    for (position, rate, expected_sign) in [
        (0.0, 1.0, 0),
        (0.3, 0.0, -1),
        (-0.3, 0.0, 1),
        (0.3, -100.0, 0),
    ] {
        let mut builder = ModelBuilder::new();
        let b = body(&mut builder, 2.0);
        let j = builder
            .add_joint_d6(BodyId::NONE, b, Transform::IDENTITY, Transform::IDENTITY)
            .unwrap();
        let mut lower = [f64::NEG_INFINITY; 6];
        let mut upper = [f64::INFINITY; 6];
        lower[0] = -0.2;
        upper[0] = 0.2;
        builder
            .set_joint_compliance(j, softness(8.0, 1.0), &lower, &upper)
            .unwrap();
        let mut model = builder.build().unwrap();
        model.gravity.host_mut().unwrap().fill(Vec3::ZERO);
        let mut state = model.state().unwrap();
        state.joint_q.host_mut().unwrap()[0] = position;
        state.joint_qd.host_mut().unwrap()[0] = rate;
        advance(&model, &mut state, 1e-4, 1, 1);
        let change = state.joint_qd.host().unwrap()[0] - rate;
        if expected_sign == 0 {
            near(change, 0.0, 1e-10);
        } else {
            assert!(change * expected_sign as f64 > 0.0);
        }
    }
}

fn floating_pair(ratio: f64, reversed: bool) -> (Model, JointId) {
    let mut builder = ModelBuilder::new();
    let parent = body(&mut builder, 2.0);
    let child = body(&mut builder, 2.0 * ratio);
    builder
        .add_joint_free(
            BodyId::NONE,
            parent,
            Transform::IDENTITY,
            Transform::IDENTITY,
        )
        .unwrap();
    let j = builder
        .add_joint_d6(parent, child, Transform::IDENTITY, Transform::IDENTITY)
        .unwrap();
    builder.set_joint_reversed(j, reversed).unwrap();
    builder
        .set_joint_compliance(j, softness(8.0, 1.0), &[0.0; 6], &[0.0; 6])
        .unwrap();
    let mut model = builder.build().unwrap();
    model.gravity.host_mut().unwrap().fill(Vec3::ZERO);
    (model, j)
}

#[test]
fn floating_pair_uses_reduced_mass_and_preserves_linear_momentum() {
    for reversed in [false, true] {
        for ratio in [0.001, 1.0, 1000.0] {
            let (model, joint) = floating_pair(ratio, reversed);
            let c = model.joint_coord_offset.host().unwrap()[joint.index()] as usize;
            let d = model.joint_dof_offset.host().unwrap()[joint.index()] as usize;
            let mut state = model.state().unwrap();
            state.joint_q.host_mut().unwrap()[c] = 0.2;
            let dt = 1.0 / 960.0;
            advance(&model, &mut state, dt, 1, 1);
            let omega = std::f64::consts::TAU * 8.0;
            near(
                state.joint_qd.host().unwrap()[d],
                -omega * omega * 0.2 * dt / (1.0 + omega * dt).powi(2),
                1e-8,
            );
            let velocities = state.body_qd.host().unwrap();
            assert!(
                (velocities[0].linear * 2.0 + velocities[1].linear * 2.0 * ratio).length() < 1e-8
            );
            advance(&model, &mut state, 1.0 / 120.0, 120, 8);
            near(state.joint_q.host().unwrap()[c], 0.0, 1e-7);
        }
    }
}

#[test]
fn builder_validates_atomically_and_hard_limits_replace_springs() {
    let mut builder = ModelBuilder::new();
    let b = body(&mut builder, 2.0);
    let j = builder
        .add_joint_d6(BodyId::NONE, b, Transform::IDENTITY, Transform::IDENTITY)
        .unwrap();
    builder
        .set_joint_compliance(j, softness(8.0, 1.0), &[0.0; 6], &[0.0; 6])
        .unwrap();
    for s in [
        softness(-1.0, 1.0),
        softness(1.0, -1.0),
        softness(f64::NAN, 1.0),
        softness(1.0, f64::INFINITY),
        softness(f64::MAX, 1.0),
    ] {
        assert!(
            builder
                .set_joint_compliance(j, s, &[0.0; 6], &[0.0; 6])
                .is_err()
        );
    }
    for bound in [f64::NAN, f64::INFINITY, 1.0] {
        assert!(
            builder
                .set_joint_compliance(j, softness(30.0, 1.0), &[bound; 6], &[0.0; 6])
                .is_err()
        );
    }
    let model = builder.build().unwrap();
    assert!(
        model
            .joint_compliance
            .host()
            .unwrap()
            .iter()
            .all(|c| c.natural_frequency == 8.0)
    );
    let mut builder = ModelBuilder::new();
    let b = body(&mut builder, 2.0);
    let j = builder
        .add_joint_d6(BodyId::NONE, b, Transform::IDENTITY, Transform::IDENTITY)
        .unwrap();
    builder
        .set_joint_compliance(j, softness(8.0, 1.0), &[0.0; 6], &[0.0; 6])
        .unwrap();
    builder.set_joint_limits(j, &[0.0; 6], &[0.0; 6]).unwrap();
    assert!(
        builder
            .build()
            .unwrap()
            .joint_compliance
            .host()
            .unwrap()
            .iter()
            .all(|c| c.enabled == 0)
    );
    let mut builder = ModelBuilder::new();
    let b = body(&mut builder, 2.0);
    let j = builder
        .add_joint_free(BodyId::NONE, b, Transform::IDENTITY, Transform::IDENTITY)
        .unwrap();
    assert!(
        builder
            .set_joint_compliance(j, softness(8.0, 1.0), &[0.0; 6], &[0.0; 6])
            .is_err()
    );
}

#[test]
fn invalid_mutated_parameters_return_errors_without_corrupting_input() {
    for bounds in [[1.0, -1.0], [f64::NAN, 0.0], [0.0, f64::NEG_INFINITY]] {
        let mut model = model(1.0, 8.0, 1.0);
        let c = &mut model.joint_compliance.host_mut().unwrap()[0];
        c.lower = bounds[0];
        c.upper = bounds[1];
        let state = model.state().unwrap();
        let mut out = model.state().unwrap();
        assert!(
            FeatherstoneSolver::new_cpu()
                .step(
                    &model,
                    &state,
                    &model.control(),
                    &Contacts::with_capacity(0),
                    0.01,
                    &mut out
                )
                .is_err()
        );
        assert!(state.joint_q.host().unwrap().iter().all(|q| q.is_finite()));
        assert!(
            state
                .body_f
                .host()
                .unwrap()
                .iter()
                .all(|f| *f == SpatialVector::ZERO)
        );
    }
}
