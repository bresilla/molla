use molla_core::{BodyId, WorldId};
use molla_math::{Mat3, Quat, SpatialInertia, SpatialVector, Transform, Vec3};
use molla_sim::{BodyParams, Contacts, ModelBuilder};
use molla_solvers::rnea::inverse_dynamics_wrenches;
use molla_solvers::{FeatherstoneSolver, Solver};

fn near(actual: SpatialVector, expected: SpatialVector, tolerance: f64) {
    assert!(
        (actual.linear - expected.linear).length() < tolerance,
        "actual={actual:?} expected={expected:?}"
    );
    assert!(
        (actual.angular - expected.angular).length() < tolerance,
        "actual={actual:?} expected={expected:?}"
    );
}

#[test]
fn disabled_motion_subspace_ignores_latent_rates_and_accelerations() {
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), -Vec3::Y * 10.0);
    let child = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass: 2.0,
        com: Vec3::X,
        inertia: Mat3::IDENTITY,
    }));
    let joint = builder
        .add_joint_revolute(
            BodyId::NONE,
            child,
            Transform::IDENTITY,
            Transform::IDENTITY,
            Vec3::Z,
        )
        .unwrap();
    builder.set_joint_enabled(joint, false).unwrap();
    let model = builder.build().unwrap();
    let mut state = model.state().unwrap();
    state.joint_qd.host_mut().unwrap()[0] = 3.0;
    molla_sim::eval_fk(&model, &mut state).unwrap();
    let result =
        inverse_dynamics_wrenches(&model, &state, 0, &[4.0], &[SpatialVector::ZERO]).unwrap();
    assert_eq!(result.efforts, [0.0]);
    near(
        result.joints[0],
        SpatialVector::new(Vec3::Y * 20.0, Vec3::Z * 20.0),
        1e-12,
    );
}

#[test]
fn fixed_chain_reports_anchor_wrenches_including_external_com_loads() {
    for translation in [Vec3::ZERO, Vec3::new(1e6, -2e6, 3e6)] {
        let mut builder = ModelBuilder::new();
        let gravity = Vec3::new(0.0, -10.0, 0.0);
        builder.set_world_gravity(WorldId(0), gravity);
        builder.begin_articulation();
        let a = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 2.0,
            com: Vec3::new(0.25, 0.5, 0.0),
            inertia: Mat3::IDENTITY,
        }));
        let b = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 3.0,
            com: Vec3::new(0.5, 0.0, 0.25),
            inertia: Mat3::IDENTITY,
        }));
        builder
            .add_joint_fixed(
                BodyId::NONE,
                a,
                Transform::from_translation(translation),
                Transform::IDENTITY,
            )
            .unwrap();
        builder
            .add_joint_fixed(
                a,
                b,
                Transform::from_translation(Vec3::X * 2.0),
                Transform::from_translation(Vec3::Y * 0.5),
            )
            .unwrap();
        let model = builder.build().unwrap();
        let mut state = model.state().unwrap();
        molla_sim::eval_fk(&model, &mut state).unwrap();
        let external = [
            SpatialVector::new(Vec3::Z * 4.0, Vec3::X),
            SpatialVector::new(Vec3::X * 7.0, Vec3::Y * 2.0),
        ];
        let samples = inverse_dynamics_wrenches(&model, &state, 0, &[], &external).unwrap();
        assert!(samples.efforts.is_empty());
        assert_eq!(samples.joints.len(), 2);
        let positions = state.body_q.host().unwrap();
        let centers = [
            positions[0].position + Vec3::new(0.25, 0.5, 0.0),
            positions[1].position + Vec3::new(0.5, 0.0, 0.25),
        ];
        let force_a = -gravity * 2.0 - external[0].linear;
        let force_b = -gravity * 3.0 - external[1].linear;
        let anchor_b = translation + Vec3::X * 2.0;
        let required_b = SpatialVector::new(
            force_b,
            (centers[1] - anchor_b).cross(force_b) - external[1].angular,
        );
        let required_a = SpatialVector::new(
            force_a + force_b,
            (centers[0] - translation).cross(force_a) - external[0].angular
                + (centers[1] - translation).cross(force_b)
                - external[1].angular,
        );
        near(samples.joints[0], required_a, 1e-10);
        near(samples.joints[1], required_b, 1e-10);
    }
}

#[test]
fn ball_wrench_matches_offset_com_full_inertia_newton_euler_oracle() {
    let frame = Quat::from_rotation_y(0.4) * Quat::from_rotation_x(-0.7);
    let child_frame = Transform {
        position: Vec3::new(0.2, -0.1, 0.3),
        rotation: Quat::from_rotation_z(0.6),
    };
    let inertia = Mat3::from_cols(
        Vec3::new(2.0, 0.3, -0.1),
        Vec3::new(0.3, 3.0, 0.2),
        Vec3::new(-0.1, 0.2, 4.0),
    );
    let com = Vec3::new(0.4, -0.8, 0.1);
    let gravity = Vec3::new(0.3, -9.81, -0.7);
    for translation in [Vec3::ZERO, Vec3::new(1024.0, -2048.0, 4096.0)] {
        let mut builder = ModelBuilder::new();
        builder.set_world_gravity(WorldId(0), gravity);
        let body = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 2.5,
            com,
            inertia,
        }));
        builder
            .add_joint_ball(
                BodyId::NONE,
                body,
                Transform {
                    position: translation,
                    rotation: frame,
                },
                child_frame,
            )
            .unwrap();
        let model = builder.build().unwrap();
        let mut state = model.state().unwrap();
        state
            .joint_qd
            .host_mut()
            .unwrap()
            .copy_from_slice(&[0.6, -0.3, 0.9]);
        molla_sim::eval_fk(&model, &mut state).unwrap();
        let alpha = frame * Vec3::new(-0.4, 0.7, 0.2);
        let omega = frame * Vec3::new(0.6, -0.3, 0.9);
        let pose = state.body_q.host().unwrap()[0];
        let arm = pose.rotation * (com - child_frame.position);
        let rotation = Mat3::from_quat(pose.rotation);
        let world_inertia = rotation * inertia * rotation.transpose();
        let external = SpatialVector::new(Vec3::new(1.0, -2.0, 3.0), Vec3::new(-0.7, 0.3, 0.8));
        let force =
            (alpha.cross(arm) + omega.cross(omega.cross(arm)) - gravity) * 2.5 - external.linear;
        let torque = world_inertia * alpha + omega.cross(world_inertia * omega) - external.angular
            + arm.cross(force);
        let samples =
            inverse_dynamics_wrenches(&model, &state, 0, &[-0.4, 0.7, 0.2], &[external]).unwrap();
        near(samples.joints[0], SpatialVector::new(force, torque), 1e-9);
        let generalized = frame.conjugate() * torque;
        for (actual, expected) in samples.efforts.iter().zip(generalized.to_array()) {
            assert!((actual - expected).abs() < 1e-9);
        }
    }
}

#[test]
fn free_body_requires_no_joint_wrench_at_its_applied_acceleration() {
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), -Vec3::Y * 10.0);
    let body = builder.add_body(BodyParams::new().with_mass(2.0));
    builder
        .add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
        .unwrap();
    let model = builder.build().unwrap();
    let mut state = model.state().unwrap();
    molla_sim::eval_fk(&model, &mut state).unwrap();
    let external = SpatialVector::new(Vec3::X * 6.0, Vec3::Z * 4.0);
    let samples = inverse_dynamics_wrenches(
        &model,
        &state,
        0,
        &[3.0, -10.0, 0.0, 0.0, 0.0, 4.0],
        &[external],
    )
    .unwrap();
    near(samples.joints[0], SpatialVector::ZERO, 1e-10);
    assert!(samples.efforts.iter().all(|effort| effort.abs() < 1e-10));
    assert!(inverse_dynamics_wrenches(&model, &state, 0, &[0.0; 6], &[]).is_err());
    assert!(inverse_dynamics_wrenches(&model, &state, 0, &[f64::NAN; 6], &[external]).is_err());
    assert!(inverse_dynamics_wrenches(&model, &state, 0, &[0.0; 5], &[external]).is_err());
    assert!(inverse_dynamics_wrenches(&model, &state, 1, &[0.0; 6], &[external]).is_err());
    let mut invalid = external;
    invalid.angular.x = f64::INFINITY;
    assert!(inverse_dynamics_wrenches(&model, &state, 0, &[0.0; 6], &[invalid]).is_err());
    state.body_q.host_mut().unwrap()[0].position.x = f64::NAN;
    assert!(inverse_dynamics_wrenches(&model, &state, 0, &[0.0; 6], &[external]).is_err());
}

#[test]
fn solved_scalar_stop_reports_transmission_without_a_motor_command() {
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), Vec3::ZERO);
    builder.begin_articulation();
    let parent = builder.add_body(
        BodyParams::new()
            .with_mass(2.0)
            .with_inertia(Mat3::IDENTITY * 2.0),
    );
    let child = builder.add_body(
        BodyParams::new()
            .with_mass(3.0)
            .with_inertia(Mat3::IDENTITY * 3.0),
    );
    builder
        .add_joint_free(
            BodyId::NONE,
            parent,
            Transform::IDENTITY,
            Transform::IDENTITY,
        )
        .unwrap();
    let joint = builder
        .add_joint_revolute(
            parent,
            child,
            Transform::IDENTITY,
            Transform::IDENTITY,
            Vec3::X,
        )
        .unwrap();
    builder.set_joint_limits(joint, &[0.0], &[0.0]).unwrap();
    let model = builder.build().unwrap();
    let mut input = model.state().unwrap();
    molla_sim::eval_fk(&model, &mut input).unwrap();
    input.body_f.host_mut().unwrap()[child.index()] =
        SpatialVector::new(Vec3::ZERO, Vec3::X * 10.0);
    let mut output = model.state().unwrap();
    let mut solver = FeatherstoneSolver::new_cpu().with_substeps(1);
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
    let acceleration: Vec<_> = output
        .joint_qd
        .host()
        .unwrap()
        .iter()
        .zip(input.joint_qd.host().unwrap())
        .map(|(a, b)| (a - b) / 0.01)
        .collect();
    let samples = inverse_dynamics_wrenches(
        &model,
        &input,
        0,
        &acceleration,
        input.body_f.host().unwrap(),
    )
    .unwrap();
    near(samples.joints[0], SpatialVector::ZERO, 1e-10);
    near(
        samples.joints[1],
        SpatialVector::new(Vec3::ZERO, -Vec3::X * 4.0),
        1e-10,
    );
    assert!(
        solver
            .joint_efforts()
            .iter()
            .all(|effort| effort.drive == 0.0)
    );
}

#[test]
fn joint_wrench_indices_are_local_to_the_requested_articulation() {
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), -Vec3::Y * 10.0);
    for (index, mass) in [2.0, 5.0].into_iter().enumerate() {
        builder.begin_articulation();
        let body = builder.add_body(BodyParams::new().with_mass(mass));
        builder
            .add_joint_fixed(
                BodyId::NONE,
                body,
                Transform::from_translation(Vec3::X * index as f64),
                Transform::IDENTITY,
            )
            .unwrap();
    }
    let model = builder.build().unwrap();
    let mut state = model.state().unwrap();
    molla_sim::eval_fk(&model, &mut state).unwrap();
    for (articulation, force) in [20.0, 50.0].into_iter().enumerate() {
        let samples =
            inverse_dynamics_wrenches(&model, &state, articulation, &[], &[SpatialVector::ZERO; 2])
                .unwrap();
        assert_eq!(samples.joints.len(), 1);
        near(
            samples.joints[0],
            SpatialVector::new(Vec3::Y * force, Vec3::ZERO),
            1e-10,
        );
    }
}
